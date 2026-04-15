from __future__ import annotations

import argparse

from src.config import DEFAULT_BACKBONE_LR, DEFAULT_HEAD_LR, DEFAULT_TRAIN_LR, get_spec
from src.train_lib import RuntimeOptions, make_spec, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Oxford-IIIT Pet classification experiment.")
    parser.add_argument("--preset", default="", help="Canonical preset key, e.g. densenet121_scratch.")
    parser.add_argument("--model", choices=["densenet121", "resnext50_32x4d"], default="densenet121")
    parser.add_argument("--init-mode", choices=["scratch", "finetune"], default="scratch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-only-epochs", type=int, default=3)
    parser.add_argument("--train-lr", type=float, default=DEFAULT_TRAIN_LR)
    parser.add_argument("--head-lr", type=float, default=DEFAULT_HEAD_LR)
    parser.add_argument("--backbone-lr", type=float, default=DEFAULT_BACKBONE_LR)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-root", default="task2/data")
    parser.add_argument("--output-root", default="task2/runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume-dir", default="")
    parser.add_argument("--extra-epochs", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Override the epoch budget to 1 for quick validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_dir:
        spec = make_spec(
            model_name=args.model,
            init_mode=args.init_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            seed=args.seed,
            head_only_epochs=args.head_only_epochs,
            train_lr=args.train_lr,
            head_lr=args.head_lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
        )
    elif args.preset:
        spec = get_spec(args.preset)
    else:
        spec = make_spec(
            model_name=args.model,
            init_mode=args.init_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            seed=args.seed,
            head_only_epochs=args.head_only_epochs,
            train_lr=args.train_lr,
            head_lr=args.head_lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
        )

    if args.smoke_test:
        spec = make_spec(
            model_name=spec.model_name,
            init_mode=spec.init_mode,
            epochs=1,
            batch_size=spec.batch_size,
            image_size=spec.image_size,
            seed=spec.seed,
            head_only_epochs=spec.head_only_epochs,
            train_lr=spec.train_lr,
            head_lr=spec.head_lr,
            backbone_lr=spec.backbone_lr,
            weight_decay=spec.weight_decay,
        )

    run_dir = run_experiment(
        spec,
        RuntimeOptions(
            data_root=args.data_root,
            output_root=args.output_root,
            device=args.device,
            num_workers=args.num_workers,
            resume_dir=args.resume_dir,
            extra_epochs=args.extra_epochs,
        ),
    )
    print(f"Finished run: {run_dir}")
    print(f"Results JSON: {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
