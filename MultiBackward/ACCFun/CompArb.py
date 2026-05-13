import numpy as np
import torch


class CompetenceArbiter:
    def __init__(self, num_class, tasks, eoss_temperature=1.0, proto_init_mode=False):
        if not tasks:
            raise ValueError("Task list cannot be empty for CompetenceArbiter")
        self.num_class = int(num_class)
        self.tasks = list(tasks)
        self.eoss_temperature = float(eoss_temperature)
        self.route_mode = "oracle"
        self.rank_lambda = 0.0
        self.base_temp = 1.0
        self.alpha = 1.3

        if proto_init_mode and "los" in self.tasks:
            self.weigh = {
                t: torch.ones(self.num_class, device="cpu") * (1.3 if t == "los" else 0.85)
                for t in self.tasks
            }
        else:
            self.weigh = {t: torch.ones(self.num_class, device="cpu") for t in self.tasks}
        self.max_weight_task = [self.tasks[0] for _ in range(self.num_class)]

    def update(self, all_preds_np, all_targets_np):
        targets = np.asarray(all_targets_np, dtype=int).reshape(-1)
        b = targets.shape[0]
        c = self.num_class
        if b == 0:
            return None

        counts = np.bincount(targets, minlength=c)
        self._last_counts = counts.copy()
        acc = np.zeros((len(self.tasks), c), dtype=float)
        for ti, task in enumerate(self.tasks):
            preds = np.asarray(all_preds_np.get(task, []), dtype=int)
            if preds.shape[0] != b:
                continue
            for cls in range(c):
                mask = targets == cls
                total = int(mask.sum())
                if total > 0:
                    acc[ti, cls] = float((preds[mask] == cls).sum()) / total

        mean_acc = acc.mean(axis=0)
        difficulty = np.clip(1.0 - mean_acc, 0.0, 1.0)
        high_support = counts >= 50
        boost = (1.0 + difficulty * 0.6) * (1.0 + high_support.astype(float) * 0.25)
        scores = (acc ** float(self.alpha)) * boost

        max_scores = np.where(np.isfinite(scores.max(axis=0)), scores.max(axis=0), 0.0)
        exp_scores = np.exp(scores - max_scores)
        weights = exp_scores / (exp_scores.sum(axis=0) + 1e-8)

        for ti, task in enumerate(self.tasks):
            self.weigh[task] = torch.tensor(weights[ti], dtype=torch.float32, device="cpu")
        self.max_weight_task = [self.tasks[int(np.argmax(weights[:, cls]))] for cls in range(c)]
        return None

    def cat_out(self, logits):
        if not self.tasks:
            return None
        first = logits.get(self.tasks[0], None)
        if not isinstance(first, torch.Tensor):
            return None
        device = first.device
        tau = self.eoss_temperature if self.eoss_temperature > 0 else 1.0
        out = None
        for task in self.tasks:
            if task not in logits:
                continue
            probs = torch.softmax(logits[task] / tau, dim=1)
            weights = self.weigh.get(task, torch.ones(self.num_class)).to(device)
            contrib = weights.unsqueeze(0) * probs
            out = contrib if out is None else out + contrib
        return out

    def cat_targets(self, logits, targets, epoch):
        device = targets.device
        preds = {t: torch.argmax(logits[t].to(device), dim=1) for t in self.tasks if t in logits}
        if not preds:
            return None
        task_list = list(preds.keys())
        base = preds[task_list[0]]
        same = torch.ones_like(base, dtype=torch.bool)
        for task in task_list[1:]:
            same = same & (base == preds[task])

        fused = torch.zeros_like(base)
        fused[same] = base[same]
        conflict_idx = torch.nonzero(~same).squeeze(1)
        if conflict_idx.numel() == 0:
            return torch.nn.functional.one_hot(fused, num_classes=self.num_class).float()

        counts = getattr(self, "_last_counts", None)
        for idx in conflict_idx.tolist():
            true_cls = int(targets[idx].item()) if self.route_mode == "oracle" else -1
            route_scores = []
            route_classes = []
            for task in task_list:
                route_cls = true_cls if self.route_mode == "oracle" else int(preds[task][idx].item())
                route_classes.append(route_cls)
                if route_cls < 0 or route_cls >= self.num_class:
                    route_scores.append(float("-inf"))
                    continue
                high_support = bool(counts[route_cls] >= 50) if isinstance(counts, np.ndarray) else route_cls >= 50
                temp = 0.55 if high_support else 1.0
                probs = torch.softmax(logits[task][idx] / temp, dim=0)
                conf = float(probs[route_cls].item())
                weight = float(self.weigh.get(task, torch.ones(self.num_class))[route_cls].item())
                route_scores.append(np.log(max(conf, 1e-12)) + np.log(max(weight, 1e-12)))
            best_i = int(np.argmax(np.asarray(route_scores, dtype=float)))
            best_task = task_list[best_i]
            if self.route_mode == "oracle":
                fused[idx] = preds[best_task][idx]
            else:
                fused[idx] = int(route_classes[best_i])
        return torch.nn.functional.one_hot(fused, num_classes=self.num_class).float()

    def cuda(self):
        if torch.cuda.is_available():
            for task in self.tasks:
                self.weigh[task] = self.weigh[task].cuda()
        return self

    def cpu(self):
        for task in self.tasks:
            self.weigh[task] = self.weigh[task].cpu()
        return self


Weight_acc = CompetenceArbiter

