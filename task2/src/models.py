from __future__ import annotations

from typing import List

from torch import nn
from torch.optim import AdamW, Optimizer
from torchvision.models import (
    DenseNet121_Weights,
    ResNeXt50_32X4D_Weights,
    densenet121,
    resnext50_32x4d,
)

from .config import InitMode, ModelName, NUM_CLASSES


def build_model(model_name: ModelName, init_mode: InitMode, num_classes: int = NUM_CLASSES) -> nn.Module:
    if model_name == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if init_mode == "finetune" else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_name == "resnext50_32x4d":
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if init_mode == "finetune" else None
        model = resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def classifier_head(model: nn.Module, model_name: ModelName) -> nn.Module:
    if model_name == "densenet121":
        return model.classifier
    if model_name == "resnext50_32x4d":
        return model.fc
    raise ValueError(f"Unsupported model: {model_name}")


def set_backbone_trainable(model: nn.Module, model_name: ModelName, trainable: bool) -> None:
    head = classifier_head(model, model_name)
    for param in model.parameters():
        param.requires_grad = trainable
    for param in head.parameters():
        param.requires_grad = True


def build_optimizer(
    model: nn.Module,
    model_name: ModelName,
    init_mode: InitMode,
    *,
    phase: str,
    train_lr: float,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> Optimizer:
    head_module = classifier_head(model, model_name)
    head_params = list(head_module.parameters())

    if init_mode == "scratch":
        return AdamW(model.parameters(), lr=train_lr, weight_decay=weight_decay)

    if phase == "head":
        return AdamW(head_params, lr=head_lr, weight_decay=weight_decay)

    head_ids = {id(param) for param in head_params}
    backbone_params = [param for param in model.parameters() if id(param) not in head_ids]
    return AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def num_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def learning_rates(optimizer: Optimizer) -> List[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]

