from __future__ import print_function

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from progress.bar import Bar
from torch.optim import lr_scheduler

from datasets.Cifar100LT import get_cifar100
from losses.bcl import BCLLoss
from losses.tailspec import TailSpecLoss
from models.resnet import resnet34
from MultiBackward.ACCFun.CompArb import CompetenceArbiter
from MultiBackward.GradFun.GES import GES
from train.validate import fuse_multi_outputs, valid_comparb
from utils.accuracy import AverageMeter, accuracy
from utils.common import hms_string
from utils.config import parse_args, reproducibility
from utils.logger import logger


args = parse_args()
reproducibility(args.seed)
if getattr(args, "deterministic", False):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
args.logger = logger(args)

best_acc = 0.0
many_best, med_best, few_best = 0.0, 0.0, 0.0


def _internal_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _make_losses(args, class_counts):
    losses = {}
    if "tailspec" in args.tasks:
        losses["tailspec"] = TailSpecLoss(
            cls_num_list=class_counts,
            num_classes=args.num_class,
            reweight_epoch=args.tailspec_reweight_epoch,
            max_m=args.tailspec_max_m,
            s=args.tailspec_s,
            temperature=args.tailspec_temperature,
            weight=args.tailspec_weight,
        ).cuda()
    if "bcl" in args.tasks:
        losses["bcl"] = BCLLoss(cls_num_list=class_counts, temperature=args.bcl_temperature).cuda()
    if "los" in args.tasks:
        losses["los"] = nn.CrossEntropyLoss(label_smoothing=args.label_smooth).cuda()
    return losses


def _task_loss(task, criterion, logits, features, targets, head, epoch):
    if task == "tailspec":
        return criterion(
            logits=logits,
            targets=targets,
            epoch=epoch,
            reduction="mean",
            features=features,
            centers=head.weight,
        )
    if task == "bcl":
        return criterion(head.weight, logits, features, targets)
    return criterion(logits, targets)


def _validate_loaded_los(args, model, testloader, train_class_counts):
    internal = _internal_model(model)
    if "los" not in internal.fc_heads:
        return
    args.logger("Running full validation for the loaded Stage 1 head ('los')...", level=1)
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    top1 = AverageMeter()
    correct = torch.zeros(args.num_class).cuda()
    total = torch.zeros(args.num_class).cuda()
    with torch.no_grad():
        for data_tuple in testloader:
            inputs = data_tuple[0].cuda(non_blocking=True)
            targets = data_tuple[1].cuda(non_blocking=True)
            features = internal.encoder(inputs)
            outputs = internal.fc_heads["los"](features)
            loss = criterion(outputs, targets)
            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            pred = torch.argmax(outputs, dim=1)
            ok = pred == targets
            for cls in range(args.num_class):
                mask = targets == cls
                total[cls] += mask.sum()
                correct[cls] += (ok & mask).sum()

    class_acc = correct / (total + 1e-6)
    counts = torch.tensor(train_class_counts).cuda()
    many = class_acc[counts > 100].mean().item()
    med = class_acc[(counts <= 100) & (counts >= 20)].mean().item()
    few = class_acc[counts < 20].mean().item()
    args.logger("Loaded Stage 1 Head ('los') Performance: ", level=1)
    args.logger(f"[Validation] Loss: {losses.avg:.4f}\tAcc: {top1.avg:.4f}% ", level=2)
    args.logger(f"[Stats] Many: {many * 100:.2f}%\tMedium: {med * 100:.2f}%\tFew: {few * 100:.2f}%", level=2)


def load_model(args, model, testloader, train_class_counts):
    pth = args.pretrained_pth
    print(f"Loading model from: {pth}")
    loaded = torch.load(pth, map_location=lambda storage, loc: storage)
    old_state = loaded.state_dict() if isinstance(loaded, nn.Module) else loaded
    if not isinstance(old_state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(loaded)}")

    internal = _internal_model(model)
    current = internal.state_dict()
    mapped = {}
    backbone_prefix = {
        "conv1": "encoder.0",
        "bn1": "encoder.1",
        "layer1": "encoder.3",
        "layer2": "encoder.4",
        "layer3": "encoder.5",
        "layer4": "encoder.6",
    }

    print("Applying mapping/matching from loaded state_dict keys...")
    los_mapped = False
    direct = 0
    prefix_count = 0
    for key, value in old_state.items():
        clean_key = key.replace("module.", "", 1)
        if clean_key.startswith("fc_heads.default.") and "los" in args.tasks:
            suffix = clean_key.split(".")[-1]
            new_key = f"fc_heads.los.{suffix}"
            if new_key in current and current[new_key].shape == value.shape:
                print(f"  Info: Mapping Stage 1 'fc_heads.default.{suffix}' to Stage 2 'fc_heads.los.{suffix}'.")
                mapped[new_key] = value
                los_mapped = True
            continue
        if clean_key.startswith("fc."):
            continue

        mapped_key = None
        for old_prefix, new_prefix in backbone_prefix.items():
            if clean_key.startswith(old_prefix + ".") or clean_key == old_prefix:
                mapped_key = new_prefix + clean_key[len(old_prefix):]
                break
        if mapped_key and mapped_key in current and current[mapped_key].shape == value.shape:
            mapped[mapped_key] = value
            prefix_count += 1
        elif clean_key in current and current[clean_key].shape == value.shape:
            mapped[clean_key] = value
            direct += 1

    print("\nAttempting to load processed state dict...")
    internal.load_state_dict(mapped, strict=False)
    print("State dict loaded into model using strict=False.")
    if los_mapped:
        print("  Info: Successfully mapped Stage 1 'fc_heads.default.*' to Stage 2 'fc_heads.los.*'.")
    if prefix_count:
        print(f"  Info: Mapped {prefix_count} backbone keys using prefix rules.")
    if direct:
        print(f"  Info: Matched {direct} keys directly.")

    model.cuda()
    _validate_loaded_los(args, model, testloader, train_class_counts)
    return model


def train_stage2(args, model, trainloader, valloader, testloader, train_class_counts):
    global best_acc, many_best, med_best, few_best

    internal = _internal_model(model)
    loss_functions = _make_losses(args, train_class_counts)
    params = []
    for task in args.tasks:
        params.extend(list(internal.fc_heads[task].parameters()))
    params.extend(list(internal.encoder[-1].parameters()))
    optimizer = optim.SGD(params, lr=args.finetune_lr, momentum=0.9, weight_decay=args.finetune_wd)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.finetune_epoch, eta_min=0.0)

    arbiter = CompetenceArbiter(
        num_class=args.num_class,
        tasks=args.tasks,
        eoss_temperature=args.comparb_temperature,
        proto_init_mode=False,
    ).cuda()
    arbiter.rank_lambda = args.comparb_rank_lambda
    arbiter.route_mode = args.comparb_route_mode
    arbiter.base_temp = args.comparb_base_temp
    arbiter.alpha = args.comparb_alpha

    solver = GES(
        optimizer=optimizer,
        reduction="mean",
        pc_beta_conf=args.ges_beta_conf,
        pc_scale_min=args.ges_scale_min,
        pc_scale_max=args.ges_scale_max,
        pc_boost=args.ges_boost,
        pc_boost_thresh_k=args.ges_boost_thresh_k,
        pc_scale_absolute_max=args.ges_scale_absolute_max,
    )
    args.logger("Multi-task solver: ges", level=1)
    args.logger("Inference fusion: comparb", level=1)

    best_model_state = None
    best_epoch = -1
    stagnate_epochs = 0
    prev_val_acc = None
    start_time = time.time()
    bar = Bar("Training", max=args.finetune_epoch)

    for epoch in range(args.finetune_epoch):
        internal.fc_heads.train()
        losses_meter = {task: AverageMeter() for task in args.tasks}
        top1_meter = AverageMeter()

        for batch_idx, data_tuple in enumerate(trainloader):
            inputs = data_tuple[0].cuda(non_blocking=True)
            targets = data_tuple[1].cuda(non_blocking=True)
            features = internal.encoder(inputs)
            logits = {task: internal.fc_heads[task](features) for task in args.tasks}

            losses = {}
            ok = True
            for task in args.tasks:
                loss_val = _task_loss(
                    task,
                    loss_functions[task],
                    logits[task],
                    features,
                    targets,
                    internal.fc_heads[task],
                    epoch,
                )
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    ok = False
                    break
                losses[task] = loss_val
            if not ok:
                continue

            solver.zero_grad()
            ordered_losses = [losses[t] for t in args.tasks]
            solver.pc_backward(ordered_losses)
            solver.step()

            for task, loss_val in losses.items():
                losses_meter[task].update(loss_val.item(), inputs.size(0))
            with torch.no_grad():
                fused_scores, _ = fuse_multi_outputs(
                    logits,
                    arbiter,
                    targets=targets,
                    epoch=epoch,
                    fusion_mode=args.fusion_mode,
                )
                if fused_scores is not None:
                    prec1 = accuracy(fused_scores, targets, topk=(1,))[0]
                    top1_meter.update(prec1.item(), inputs.size(0))

        val_loss, val_acc, val_cls = valid_comparb(
            valloader,
            model,
            arbiter,
            nn.CrossEntropyLoss(),
            train_class_counts,
            num_class=args.num_class,
            mode=f"Validation Multi ({args.fusion_mode})",
            epoch=epoch,
            fusion_mode=args.fusion_mode,
            update_weights=args.update_comparb_weights,
        )

        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        val_acc_percent = val_acc * 100.0
        improved = best_acc <= val_acc_percent
        if improved:
            best_acc = val_acc_percent
            many_best = val_cls[0] * 100.0
            med_best = val_cls[1] * 100.0
            few_best = val_cls[2] * 100.0
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            stagnate_epochs = 0
        else:
            stagnate_epochs += 1
        val_decline = prev_val_acc is not None and val_acc_percent < prev_val_acc
        prev_val_acc = val_acc_percent

        args.logger(f"Epoch: [{epoch + 1} | {args.finetune_epoch}]", level=1)
        train_loss_log = "\t".join([f"{task}: {meter.avg:.4f}" for task, meter in losses_meter.items()])
        args.logger(f"[Train]\tLosses: {train_loss_log}\tAcc:\t{top1_meter.avg:.4f}", level=2)
        args.logger(f"[Val  ]\tLoss:\t{val_loss:.4f}\tAcc:\t{val_acc_percent:.4f}", level=2)
        args.logger(f"[Stats]\tMany:\t{val_cls[0] * 100:.2f}\tMedium:\t{val_cls[1] * 100:.2f}\tFew:\t{val_cls[2] * 100:.2f}", level=2)
        args.logger(f"[Best ]\tAcc:\t{best_acc:.4f}\tMany:\t{many_best:.2f}\tMedium:\t{med_best:.2f}\tFew:\t{few_best:.2f}", level=2)
        args.logger(f"[Param]\tLR:\t{lr:.8f}", level=2)
        bar.next()

        if args.internal_early_stop and stagnate_epochs >= args.internal_stop_patience and val_decline:
            args.logger(
                f"[EARLY-STOP] Internal early stop triggered: best stagnated >= {args.internal_stop_patience} epochs "
                "and validation accuracy started decreasing.",
                level=1,
            )
            break

    bar.finish()
    elapsed = time.time() - start_time
    suffix = "_".join(args.losses)
    model_path = os.path.join(args.out, f"best_model_stage2_MT_{suffix}.pth")
    if best_model_state is not None:
        torch.save(best_model_state, model_path)
        args.logger(f"Saved best model state_dict to {model_path}", level=1)

    metrics = {
        "best_bacc": float(best_acc),
        "many": float(many_best),
        "medium": float(med_best),
        "few": float(few_best),
        "best_epoch": int(best_epoch),
        "tasks": list(args.tasks),
        "losses": list(args.losses),
        "multi_solver": "ges",
        "fusion_mode": args.fusion_mode,
        "comparb_route_mode": args.comparb_route_mode,
        "tailspec_reweight_epoch": int(args.tailspec_reweight_epoch),
        "tailspec_temperature": float(args.tailspec_temperature),
        "tailspec_max_m": float(args.tailspec_max_m),
        "tailspec_s": float(args.tailspec_s),
        "tailspec_weight": float(args.tailspec_weight),
        "ges_beta_conf": float(args.ges_beta_conf),
        "ges_scale_min": float(args.ges_scale_min),
        "ges_scale_max": float(args.ges_scale_max),
        "ges_boost": float(args.ges_boost),
        "ges_boost_thresh_k": float(args.ges_boost_thresh_k),
        "ges_scale_absolute_max": float(args.ges_scale_absolute_max),
        "model_path": model_path,
    }
    with open(os.path.join(args.out, "metrics_stage2.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    args.logger(f"Finish Training Stage 2 (MT - Losses: {args.losses})...", level=1)
    args.logger("Final performance...", level=1)
    args.logger(f"best bAcc (test):\t{best_acc:.4f}", level=2)
    args.logger(f"best statistics:\tMany:\t{many_best:.2f}\tMed:\t{med_best:.2f}\tFew:\t{few_best:.2f}", level=2)
    args.logger(f"Training Time: {hms_string(elapsed)}", level=1)


def main():
    print("==> Preparing imbalanced cifar100")
    trainset, valset, testset = get_cifar100(
        os.path.join(args.data_dir, "cifar100/"),
        args,
        val_ratio=args.val_ratio,
        val_from_train_ratio=args.val_from_train_ratio,
        balanced_val=args.balanced_val,
        val_samples_per_class=args.val_samples_per_class,
    )
    train_class_counts = trainset.img_num_list

    loader_common = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=bool(args.persistent_workers and args.workers > 0),
    )
    if args.workers > 0:
        loader_common["prefetch_factor"] = max(1, int(args.prefetch_factor))

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **loader_common)
    valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, **loader_common)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, **loader_common)

    print(f"==> creating resnet34 with heads: {args.tasks}")
    model = resnet34(num_classes=args.num_class, pool_size=4, task_list=args.tasks).cuda()
    if args.gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = load_model(args, model, testloader, train_class_counts)
    args.logger("Running initial validation after loading model (using CompArb)...", level=1)
    temp_arbiter = CompetenceArbiter(
        num_class=args.num_class,
        tasks=args.tasks,
        eoss_temperature=args.comparb_temperature,
        proto_init_mode=False,
    ).cuda()
    temp_arbiter.route_mode = args.comparb_route_mode
    temp_arbiter.rank_lambda = args.comparb_rank_lambda
    temp_arbiter.base_temp = args.comparb_base_temp
    temp_arbiter.alpha = args.comparb_alpha
    test_loss, test_acc, test_cls = valid_comparb(
        testloader,
        model,
        temp_arbiter,
        nn.CrossEntropyLoss(),
        train_class_counts,
        num_class=args.num_class,
        mode="Loaded Model Validation (CompArb Init)",
        epoch=0,
        fusion_mode=args.fusion_mode,
        update_weights=False,
    )
    args.logger("Loaded model performance (before Stage 2 training)... ", level=1)
    args.logger(f"[Validation] Loss: {test_loss:.4f}\tAcc: {test_acc * 100.0:.4f}% ", level=2)
    args.logger(f"[Stats] Many: {test_cls[0] * 100:.2f}%\tMedium: {test_cls[1] * 100:.2f}%\tFew: {test_cls[2] * 100:.2f}%", level=2)

    train_stage2(args, model, trainloader, valloader, testloader, train_class_counts)


if __name__ == "__main__":
    main()
