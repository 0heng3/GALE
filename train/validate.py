import time

import numpy as np
import torch
from progress.bar import Bar

from utils.accuracy import AverageMeter, accuracy


def calculate_group_accuracy(preds, targets, train_class_counts, num_class):
    classwise_correct = np.zeros(num_class, dtype=float)
    classwise_num = np.zeros(num_class, dtype=float)
    correct = preds == targets
    for cls in range(num_class):
        mask = targets == cls
        classwise_num[cls] = mask.sum()
        if classwise_num[cls] > 0:
            classwise_correct[cls] = correct[mask].sum()
    classwise_acc = np.divide(
        classwise_correct,
        classwise_num,
        out=np.zeros_like(classwise_correct),
        where=classwise_num != 0,
    )

    counts = np.asarray(train_class_counts)
    many = classwise_acc[counts > 100].mean() if np.any(counts > 100) else 0.0
    medium_mask = (counts <= 100) & (counts >= 20)
    medium = classwise_acc[medium_mask].mean() if np.any(medium_mask) else 0.0
    few = classwise_acc[counts < 20].mean() if np.any(counts < 20) else 0.0
    return float(many), float(medium), float(few)


def fuse_multi_outputs(output_dict, arbiter, targets=None, epoch=0, fusion_mode="comparb"):
    if not output_dict:
        return None, None
    first = next(iter(output_dict.values()))
    device = first.device
    fusion_mode = str(fusion_mode).lower()

    if fusion_mode == "avg":
        probs = torch.stack([torch.softmax(v, dim=1) for v in output_dict.values()], dim=0).mean(dim=0)
        return probs, torch.argmax(probs, dim=1)

    if fusion_mode == "weighted":
        probs = arbiter.cat_out(output_dict) if arbiter is not None else None
        if probs is None:
            return fuse_multi_outputs(output_dict, arbiter, targets, epoch, "avg")
        probs = probs.to(device)
        return probs, torch.argmax(probs, dim=1)

    if arbiter is None:
        return fuse_multi_outputs(output_dict, arbiter, targets, epoch, "avg")
    if targets is None:
        return fuse_multi_outputs(output_dict, arbiter, targets, epoch, "weighted")
    one_hot = arbiter.cat_targets(output_dict, targets, epoch)
    if one_hot is None:
        return None, None
    one_hot = one_hot.to(device)
    return one_hot, torch.argmax(one_hot, dim=1)


def valid_base(valloader, model, criterion, per_class_num, num_class=100, mode="Validation Base", test_aug=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    all_targets = []
    all_preds = []
    bar = Bar(mode, max=len(valloader))
    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(valloader):
            inputs = data_tuple[0].cuda(non_blocking=True)
            targets = data_tuple[1].cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            bar.suffix = f"({batch_idx + 1}/{len(valloader)}) top1: {top1.avg:.4f}"
            bar.next()
    bar.finish()
    targets_np = np.concatenate(all_targets)
    preds_np = np.concatenate(all_preds)
    group_acc = calculate_group_accuracy(preds_np, targets_np, per_class_num, num_class)
    return losses.avg, top1.avg, torch.tensor(group_acc)


def valid_comparb(dataloader, model, arbiter, criterion, train_class_counts, num_class,
                  mode="CompArb Valid", epoch=0, fusion_mode="comparb", update_weights=True):
    top1 = AverageMeter()
    model.eval()
    task_preds = {task: [] for task in arbiter.tasks}
    fused_preds = []
    targets_all = []
    bar = Bar(mode, max=len(dataloader))
    end = time.time()

    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(dataloader):
            inputs = data_tuple[0].cuda(non_blocking=True)
            targets = data_tuple[1].cuda(non_blocking=True)
            targets_all.append(targets.cpu().numpy())

            internal = model.module if isinstance(model, torch.nn.DataParallel) else model
            features = internal.encoder(inputs)
            output_dict = {
                task: internal.fc_heads[task](features)
                for task in arbiter.tasks
                if task in internal.fc_heads
            }
            for task in arbiter.tasks:
                if task in output_dict:
                    task_preds[task].append(torch.argmax(output_dict[task], dim=1).cpu().numpy())

            fused_scores, fused_labels = fuse_multi_outputs(
                output_dict, arbiter, targets=targets, epoch=epoch, fusion_mode=fusion_mode
            )
            if fused_labels is not None:
                fused_preds.append(fused_labels.cpu().numpy())
                correct = (fused_labels == targets).float().sum()
                top1.update(float(correct.item() * 100.0 / max(1, inputs.size(0))), inputs.size(0))

            elapsed = time.time() - end
            bar.suffix = f"({batch_idx + 1}/{len(dataloader)}) Batch: {elapsed:.3f}s | top1: {top1.avg:.4f}"
            bar.next()
            end = time.time()
    bar.finish()

    targets_np = np.concatenate(targets_all)
    if update_weights and fusion_mode in ("weighted", "comparb"):
        preds_np = {}
        ok = True
        for task in arbiter.tasks:
            if not task_preds[task]:
                ok = False
                break
            preds = np.concatenate(task_preds[task])
            if preds.shape != targets_np.shape:
                ok = False
                break
            preds_np[task] = preds
        if ok:
            print(f"Updating class-wise fusion weights based on {len(targets_np)} validation samples.")
            arbiter.update(preds_np, targets_np)
            arbiter.cuda()

    if not fused_preds:
        return float("nan"), float("nan"), (0.0, 0.0, 0.0)
    fused_np = np.concatenate(fused_preds)
    many, medium, few = calculate_group_accuracy(fused_np, targets_np, train_class_counts, num_class)
    acc = float((fused_np == targets_np).mean())
    print(f"{mode} Result ({fusion_mode}): Acc@1 {acc * 100:.3f}% Loss nan")
    print(f"Many: {many * 100:.2f}% Medium: {medium * 100:.2f}% Few: {few * 100:.2f}%")
    return float("nan"), acc, (many, medium, few)
