import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch


def parse_args(run_type="terminal"):
    parser = argparse.ArgumentParser(description="GALE Stage-II training")

    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--num_class", type=int, default=100)
    parser.add_argument("--imb_ratio", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--val_from_train_ratio", type=float, default=0.5)
    parser.add_argument("--balanced_val", action="store_true", default=True)
    parser.add_argument("--val_samples_per_class", type=int, default=100)

    parser.add_argument("--network", default="resnet34", choices=["resnet34"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cur_stage", default="stage2", choices=["stage2"])
    parser.add_argument("--finetune_epoch", type=int, default=20)
    parser.add_argument("--finetune_lr", type=float, default=0.0005)
    parser.add_argument("--finetune_wd", type=float, default=0.0)
    parser.add_argument("--label_smooth", type=float, default=0.98)

    parser.add_argument("--losses", nargs="+", default=["tailspec", "bcl", "los"],
                        choices=["tailspec", "bcl", "los"])
    parser.add_argument("--tasks", nargs="+", default=None,
                        choices=["tailspec", "bcl", "los"])
    parser.add_argument("--multi_solver", default="ges", choices=["ges"])
    parser.add_argument("--fusion_mode", default="comparb", choices=["comparb", "weighted", "avg"])

    parser.add_argument("--tailspec_reweight_epoch", type=int, default=0)
    parser.add_argument("--tailspec_temperature", type=float, default=0.1)
    parser.add_argument("--tailspec_s", type=float, default=70.0)
    parser.add_argument("--tailspec_max_m", type=float, default=0.1)
    parser.add_argument("--tailspec_weight", type=float, default=3.25)
    parser.add_argument("--bcl_temperature", type=float, default=0.1)

    parser.add_argument("--ges_beta_conf", type=float, default=0.65)
    parser.add_argument("--ges_scale_min", type=float, default=1.2)
    parser.add_argument("--ges_scale_max", type=float, default=3.0)
    parser.add_argument("--ges_boost", type=float, default=2.2)
    parser.add_argument("--ges_boost_thresh_k", type=float, default=0.6)
    parser.add_argument("--ges_scale_absolute_max", type=float, default=2.0)

    parser.add_argument("--comparb_rank_lambda", type=float, default=0.0)
    parser.add_argument("--comparb_route_mode", default="oracle",
                        choices=["oracle", "pred_conf", "pred_weighted_conf"])
    parser.add_argument("--comparb_temperature", type=float, default=1.0)
    parser.add_argument("--comparb_base_temp", type=float, default=1.0)
    parser.add_argument("--comparb_alpha", type=float, default=1.3)
    parser.add_argument("--no_update_comparb_weights", action="store_true")

    parser.add_argument("--internal_early_stop", action="store_true", default=True)
    parser.add_argument("--internal_stop_patience", type=int, default=2)
    parser.add_argument(
        "--pretrained_pth",
        default=None,
    )
    parser.add_argument("--out", default="output")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--deterministic", action="store_true", default=False)

    args = parser.parse_args()
    if args.tasks is None:
        args.tasks = list(args.losses)

    args.update_comparb_weights = not args.no_update_comparb_weights
    args.acc_fun_impl = "comparb"

    now = datetime.now()
    time_str = f"{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
    base_out_dir = args.out if args.out != "output" else f"output/{args.dataset}_IR={args.imb_ratio}_{args.cur_stage}"
    args.out = os.path.join(base_out_dir, time_str)
    os.makedirs(args.out, exist_ok=True)

    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args


def reproducibility(seed):
    if seed == "None":
        return
    seed = int(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
