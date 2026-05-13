import torch


class GES:
    def __init__(
        self,
        optimizer,
        reduction="mean",
        pc_beta_conf=0.65,
        pc_scale_min=1.2,
        pc_scale_max=3.0,
        pc_boost=2.2,
        pc_boost_thresh_k=0.6,
        pc_scale_absolute_max=2.0,
    ):
        self._optim = optimizer
        self._reduction = reduction
        self._variant = "ges"
        self._pc_beta_conf = float(pc_beta_conf)
        self._pc_scale_min = float(pc_scale_min)
        self._pc_scale_max = float(pc_scale_max)
        self._pc_boost = float(pc_boost)
        self._pc_boost_thresh_k = float(pc_boost_thresh_k)
        self._pc_scale_absolute_max = float(pc_scale_absolute_max)
        self._last_stats = {}

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def get_last_stats(self):
        return dict(self._last_stats)

    def pc_backward(self, objectives):
        grads, shapes, has_grads = self._pack_grad(objectives)
        g = torch.stack(grads, dim=0)
        eps = 1e-8
        g_scaled, extra = self._scale_by_energy(g, eps)
        merged = self._merge_grads(g_scaled, has_grads, eps)
        self._record_stats(g, g_scaled, extra)
        self._set_grad(self._unflatten_grad(merged, shapes[0]))

    def _scale_by_energy(self, g, eps):
        t = g.size(0)
        device = g.device
        norms = g.norm(dim=1).clamp_min(eps)

        abs_sorted = torch.sort(g.abs(), dim=1).values
        d = g.size(1)
        p50 = abs_sorted[:, int(max(0, min(d - 1, int(0.50 * (d - 1)))))]
        p75 = abs_sorted[:, int(max(0, min(d - 1, int(0.75 * (d - 1)))))]
        loss_vals = (0.3 * p50 + 0.7 * p75).clamp_min(eps)

        grad_to_loss = (norms / (loss_vals + eps)).clamp_min(0.0)
        grad_to_loss_n = grad_to_loss / grad_to_loss.max().clamp_min(eps)

        cos = (g @ g.t()) / (norms[:, None] * norms[None, :] + eps)
        cos.fill_diagonal_(0.0)
        neg_cos = torch.clamp(-cos, min=0.0)
        conflict_score = neg_cos.sum(dim=1) / max(1, t - 1)
        conflict_n = conflict_score / conflict_score.max().clamp_min(eps)

        importance_raw = torch.max(grad_to_loss_n, conflict_n * self._pc_beta_conf)
        importance = (importance_raw - importance_raw.min()) + eps
        importance = importance / importance.mean().clamp_min(eps)

        base = importance / importance.mean().clamp_min(eps)
        scales = base.clamp(min=self._pc_scale_min, max=self._pc_scale_max)
        thr = max(0.40, importance.mean().item() + self._pc_boost_thresh_k * importance.std().item())
        boosted = importance >= thr
        if boosted.any():
            scales[boosted] = (scales[boosted] * self._pc_boost).clamp(max=self._pc_scale_absolute_max)

        g_scaled = g * scales[:, None]
        offdiag = ~torch.eye(t, dtype=torch.bool, device=device)
        pair_cos = cos[offdiag]
        return g_scaled, {
            "num_tasks": int(t),
            "total_pairs": int(pair_cos.numel()),
            "conflict_pairs": int((pair_cos < 0).sum().item()) if pair_cos.numel() else 0,
            "projection_ops": 0,
            "final_scales": scales.detach().cpu().numpy().astype(float).tolist(),
            "importance": importance.detach().cpu().numpy().astype(float).tolist(),
        }

    def _record_stats(self, g_pre, g_post, extra):
        eps = 1e-8
        pre_norm = g_pre.norm(dim=1).clamp_min(eps)
        post_norm = g_post.norm(dim=1).clamp_min(eps)
        retention = post_norm / pre_norm
        self._last_stats = {
            "variant": "ges",
            "num_tasks": int(g_pre.size(0)),
            "pre_norm_mean": float(pre_norm.mean().item()),
            "post_norm_mean": float(post_norm.mean().item()),
            "retention_mean": float(retention.mean().item()),
            **extra,
        }

    def _merge_grads(self, g, has_grads, eps):
        device = g.device
        masks = []
        for h in has_grads:
            masks.append(h.to(device) if isinstance(h, torch.Tensor) else torch.as_tensor(h, device=device))
        masks = torch.stack(masks)
        merged = torch.zeros_like(g[0])
        if self._reduction == "sum":
            for i in range(g.size(0)):
                merged += g[i] * masks[i]
            return merged
        denom = masks.sum(dim=0).float()
        for i in range(g.size(0)):
            merged += g[i] * masks[i]
        return merged / (denom + eps)

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        self._optim.zero_grad(set_to_none=True)
        return grads, shapes, has_grads

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                shape.append(p.shape)
                if p.grad is None:
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                else:
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    @staticmethod
    def _flatten_grad(grads, shapes):
        return torch.cat([g.reshape(-1) for g, _shape in zip(grads, shapes)])

    @staticmethod
    def _unflatten_grad(grads, shapes):
        unflatten = []
        idx = 0
        for shape in shapes:
            length = int(torch.prod(torch.tensor(shape)).item())
            unflatten.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                p.grad = grads[idx]
                idx += 1

