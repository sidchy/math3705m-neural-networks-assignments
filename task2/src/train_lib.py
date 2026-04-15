from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
(PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

import torch
from torch import nn

from .config import (
    DATASET_NAME,
    DEFAULT_BACKBONE_LR,
    DEFAULT_HEAD_LR,
    DEFAULT_HEAD_ONLY_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
    DEFAULT_TRAIN_LR,
    DEFAULT_WEIGHT_DECAY,
    ExperimentSpec,
    INIT_DISPLAY_NAMES,
    MODEL_DISPLAY_NAMES,
    NUM_CLASSES,
)
from .data import DatasetBundle, build_dataloaders
from .metrics import accuracy_score, confusion_matrix, macro_f1_score, top_confused_pairs
from .models import build_model, build_optimizer, learning_rates, num_parameters, set_backbone_trainable
from .plotting import plot_history, plot_predictions_grid, plot_top_confusions


@dataclass
class RuntimeOptions:
    data_root: str
    output_root: str
    device: str = "auto"
    num_workers: int = 2
    resume_dir: str = ""
    extra_epochs: int = 0


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_torch_load(path: Path, map_location: str = "cpu") -> Dict[str, object]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _phase_for_epoch(spec: ExperimentSpec, epoch: int) -> str:
    if spec.init_mode == "finetune" and epoch <= spec.head_only_epochs:
        return "head"
    return "full"


def _build_run_dir(output_root: Path, spec: ExperimentSpec) -> Path:
    return output_root / f"{spec.model_name}_{spec.init_mode}_{_timestamp()}"


def _write_history_csv(history: Sequence[Dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "phase",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr_backbone",
        "lr_head",
        "epoch_seconds",
        "total_train_seconds",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _write_json(payload: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_items += batch_size
    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def _evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    sample_limit: int = 0,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    samples: List[Dict[str, object]] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == targets).sum().item())
            total_items += batch_size

            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

            if sample_limit > 0 and len(samples) < sample_limit:
                remaining = sample_limit - len(samples)
                batch_images = images.detach().cpu().numpy()[:remaining]
                batch_true = targets.detach().cpu().numpy()[:remaining]
                batch_pred = preds.detach().cpu().numpy()[:remaining]
                for image, truth, pred in zip(batch_images, batch_true, batch_pred):
                    samples.append({"image": image, "true": int(truth), "pred": int(pred)})

    return {
        "loss": total_loss / max(total_items, 1),
        "acc": total_correct / max(total_items, 1),
        "y_true": np.array(y_true, dtype=np.int64),
        "y_pred": np.array(y_pred, dtype=np.int64),
        "samples": samples,
    }


def _load_resume_spec_and_state(resume_dir: Path) -> Tuple[ExperimentSpec, Dict[str, object]]:
    checkpoint = _safe_torch_load(resume_dir / "last.pt", map_location="cpu")
    config = checkpoint["config"]
    spec = ExperimentSpec(
        model_name=str(config["model_name"]),
        init_mode=str(config["init_mode"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        image_size=int(config["image_size"]),
        seed=int(config["seed"]),
        head_only_epochs=int(config["head_only_epochs"]),
        train_lr=float(config["train_lr"]),
        head_lr=float(config["head_lr"]),
        backbone_lr=float(config["backbone_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    return spec, checkpoint


def run_experiment(spec: ExperimentSpec, runtime: RuntimeOptions) -> Path:
    seed_everything(spec.seed)
    device = resolve_device(runtime.device)
    output_root = Path(runtime.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint: Optional[Dict[str, object]] = None
    if runtime.resume_dir:
        run_dir = Path(runtime.resume_dir)
        loaded_spec, checkpoint = _load_resume_spec_and_state(run_dir)
        if runtime.extra_epochs <= 0:
            raise ValueError("--extra-epochs must be positive when resuming a run.")
        spec = ExperimentSpec(
            model_name=loaded_spec.model_name,
            init_mode=loaded_spec.init_mode,
            epochs=loaded_spec.epochs + runtime.extra_epochs,
            batch_size=loaded_spec.batch_size,
            image_size=loaded_spec.image_size,
            seed=loaded_spec.seed,
            head_only_epochs=loaded_spec.head_only_epochs,
            train_lr=loaded_spec.train_lr,
            head_lr=loaded_spec.head_lr,
            backbone_lr=loaded_spec.backbone_lr,
            weight_decay=loaded_spec.weight_decay,
        )
    else:
        run_dir = _build_run_dir(output_root, spec)
        run_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_dataloaders(
        runtime.data_root,
        image_size=spec.image_size,
        batch_size=spec.batch_size,
        seed=spec.seed,
        num_workers=runtime.num_workers,
    )

    model = build_model(spec.model_name, spec.init_mode, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []
    best_val_acc = float("-inf")
    best_epoch = 0
    train_seconds = 0.0
    start_epoch = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        history = list(checkpoint.get("history", []))
        best_val_acc = float(checkpoint.get("best_val_acc", float("-inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        train_seconds = float(checkpoint.get("train_seconds", 0.0))
        start_epoch = int(checkpoint.get("epoch", 0))

    model = model.to(device)

    initial_phase = _phase_for_epoch(spec, max(start_epoch, 1) if start_epoch > 0 else 1)
    if spec.init_mode == "finetune":
        set_backbone_trainable(model, spec.model_name, trainable=(initial_phase == "full"))

    optimizer = build_optimizer(
        model,
        spec.model_name,
        spec.init_mode,
        phase=initial_phase,
        train_lr=spec.train_lr,
        backbone_lr=spec.backbone_lr,
        head_lr=spec.head_lr,
        weight_decay=spec.weight_decay,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    for epoch in range(start_epoch + 1, spec.epochs + 1):
        phase = _phase_for_epoch(spec, epoch)
        if epoch == 1 and spec.init_mode == "finetune":
            set_backbone_trainable(model, spec.model_name, trainable=False)
        if spec.init_mode == "finetune" and epoch == spec.head_only_epochs + 1:
            set_backbone_trainable(model, spec.model_name, trainable=True)
            optimizer = build_optimizer(
                model,
                spec.model_name,
                spec.init_mode,
                phase="full",
                train_lr=spec.train_lr,
                backbone_lr=spec.backbone_lr,
                head_lr=spec.head_lr,
                weight_decay=spec.weight_decay,
            )

        epoch_start = time.time()
        train_loss, train_acc = _train_one_epoch(model, bundle.train_loader, optimizer, criterion, device)
        val_metrics = _evaluate(model, bundle.val_loader, criterion, device)
        epoch_seconds = time.time() - epoch_start
        train_seconds += epoch_seconds
        lr_values = learning_rates(optimizer)
        lr_backbone = lr_values[0]
        lr_head = lr_values[-1]

        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": float(val_metrics["loss"]),
                "val_acc": float(val_metrics["acc"]),
                "lr_backbone": lr_backbone,
                "lr_head": lr_head,
                "epoch_seconds": epoch_seconds,
                "total_train_seconds": train_seconds,
            }
        )

        state = {
            "epoch": epoch,
            "config": asdict(spec),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "train_seconds": train_seconds,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "class_names": bundle.class_names,
            "dataset_name": DATASET_NAME,
        }
        torch.save(state, run_dir / "last.pt")

        if float(val_metrics["acc"]) > best_val_acc:
            best_val_acc = float(val_metrics["acc"])
            best_epoch = epoch
            state["best_val_acc"] = best_val_acc
            state["best_epoch"] = best_epoch
            torch.save(state, run_dir / "best.pt")

    best_checkpoint = _safe_torch_load(run_dir / "best.pt", map_location="cpu")
    best_model = build_model(spec.model_name, spec.init_mode, num_classes=NUM_CLASSES)
    best_model.load_state_dict(best_checkpoint["model_state"])
    best_model = best_model.to(device)

    test_metrics = _evaluate(best_model, bundle.test_loader, criterion, device, sample_limit=16)
    y_true = test_metrics["y_true"]
    y_pred = test_metrics["y_pred"]
    cm = confusion_matrix(y_true, y_pred, NUM_CLASSES)
    confusions = top_confused_pairs(cm, bundle.class_names, top_k=10)
    macro_f1 = macro_f1_score(y_true, y_pred, NUM_CLASSES)

    history_csv = run_dir / "history.csv"
    _write_history_csv(history, history_csv)
    plot_history(history, run_dir / "curves.png", f"{MODEL_DISPLAY_NAMES[spec.model_name]} - {INIT_DISPLAY_NAMES[spec.init_mode]}")
    plot_predictions_grid(
        test_metrics["samples"],
        bundle.class_names,
        run_dir / "predictions_grid.png",
        f"Predictions: {MODEL_DISPLAY_NAMES[spec.model_name]} ({INIT_DISPLAY_NAMES[spec.init_mode]})",
    )
    plot_top_confusions(
        confusions,
        run_dir / "top_confusions.png",
        f"Top Confusions: {MODEL_DISPLAY_NAMES[spec.model_name]} ({INIT_DISPLAY_NAMES[spec.init_mode]})",
    )

    results = {
        "model": spec.model_name,
        "model_display_name": MODEL_DISPLAY_NAMES[spec.model_name],
        "init_mode": spec.init_mode,
        "init_display_name": INIT_DISPLAY_NAMES[spec.init_mode],
        "display_name": spec.display_name,
        "dataset_name": DATASET_NAME,
        "best_val_acc": best_val_acc,
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "train_seconds": float(train_seconds),
        "epochs": int(spec.epochs),
        "batch_size": int(spec.batch_size),
        "image_size": int(spec.image_size),
        "best_epoch": int(best_epoch),
        "num_params": int(num_parameters(best_model)),
        "train_size": bundle.train_size,
        "val_size": bundle.val_size,
        "test_size": bundle.test_size,
        "history_path": str(history_csv),
        "run_dir": str(run_dir),
        "class_names": bundle.class_names,
        "config": asdict(spec),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(results, run_dir / "results.json")
    return run_dir


def make_spec(
    *,
    model_name: str,
    init_mode: str,
    epochs: int,
    batch_size: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    head_only_epochs: int = DEFAULT_HEAD_ONLY_EPOCHS,
    train_lr: float = DEFAULT_TRAIN_LR,
    head_lr: float = DEFAULT_HEAD_LR,
    backbone_lr: float = DEFAULT_BACKBONE_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> ExperimentSpec:
    return ExperimentSpec(
        model_name=model_name,  # type: ignore[arg-type]
        init_mode=init_mode,  # type: ignore[arg-type]
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        head_only_epochs=head_only_epochs,
        train_lr=train_lr,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
    )


def load_results(result_path: Path) -> Dict[str, object]:
    return json.loads(result_path.read_text(encoding="utf-8"))
