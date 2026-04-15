from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from src.config import DEFAULT_MIN_TRAIN_SECONDS, PRESET_ORDER, canonical_specs
from src.train_lib import RuntimeOptions, load_results, make_spec, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four canonical Oxford-IIIT Pet experiments.")
    parser.add_argument("--data-root", default="task2/data")
    parser.add_argument("--output-root", default="task2/runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--min-train-seconds", type=float, default=DEFAULT_MIN_TRAIN_SECONDS)
    parser.add_argument("--extension-epochs", type=int, default=10)
    return parser.parse_args()


def _sum_train_seconds(run_dirs: Dict[str, Path]) -> float:
    total = 0.0
    for run_dir in run_dirs.values():
        payload = load_results(run_dir / "results.json")
        total += float(payload["train_seconds"])
    return total


def main() -> None:
    args = parse_args()
    runtime = RuntimeOptions(
        data_root=args.data_root,
        output_root=args.output_root,
        device=args.device,
        num_workers=args.num_workers,
    )

    run_dirs: Dict[str, Path] = {}
    for spec in canonical_specs():
        effective_spec = spec
        if args.smoke_test:
            effective_spec = make_spec(
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
        run_dirs[spec.preset_key] = run_experiment(effective_spec, runtime)

    if not args.smoke_test:
        while _sum_train_seconds(run_dirs) < args.min_train_seconds:
            for key in ("densenet121_scratch", "resnext50_32x4d_scratch"):
                run_dir = run_dirs[key]
                runtime.resume_dir = str(run_dir)
                runtime.extra_epochs = args.extension_epochs
                spec = make_spec(model_name="densenet121", init_mode="scratch", epochs=1, batch_size=64)
                run_dirs[key] = run_experiment(spec, runtime)
                runtime.resume_dir = ""
                runtime.extra_epochs = 0
                if _sum_train_seconds(run_dirs) >= args.min_train_seconds:
                    break

    manifest = {
        "runs": {key: str(path) for key, path in run_dirs.items()},
        "total_train_seconds": _sum_train_seconds(run_dirs),
    }
    manifest_path = Path(args.output_root) / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

