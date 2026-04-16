from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal

ModelName = Literal["densenet121", "resnext50_32x4d"]
InitMode = Literal["scratch", "finetune"]

DATASET_NAME = "Oxford-IIIT Pet"
NUM_CLASSES = 37
DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 224
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HEAD_ONLY_EPOCHS = 3
DEFAULT_TRAIN_LR = 3e-4
DEFAULT_HEAD_LR = 1e-3
DEFAULT_BACKBONE_LR = 1e-4
DEFAULT_MIN_TRAIN_SECONDS = 7200.0

MODEL_DISPLAY_NAMES: Dict[ModelName, str] = {
    "densenet121": "DenseNet121",
    "resnext50_32x4d": "ResNeXt50_32x4d",
}

INIT_DISPLAY_NAMES: Dict[InitMode, str] = {
    "scratch": "From Scratch",
    "finetune": "Fine-tune",
}


@dataclass(frozen=True)
class ExperimentSpec:
    model_name: ModelName
    init_mode: InitMode
    epochs: int
    batch_size: int
    image_size: int = DEFAULT_IMAGE_SIZE
    seed: int = DEFAULT_SEED
    head_only_epochs: int = DEFAULT_HEAD_ONLY_EPOCHS
    train_lr: float = DEFAULT_TRAIN_LR
    head_lr: float = DEFAULT_HEAD_LR
    backbone_lr: float = DEFAULT_BACKBONE_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY

    @property
    def preset_key(self) -> str:
        return f"{self.model_name}_{self.init_mode}"

    @property
    def display_name(self) -> str:
        return f"{MODEL_DISPLAY_NAMES[self.model_name]} ({INIT_DISPLAY_NAMES[self.init_mode]})"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


CANONICAL_PRESETS: Dict[str, ExperimentSpec] = {
    "densenet121_scratch": ExperimentSpec(
        model_name="densenet121",
        init_mode="scratch",
        epochs=30,
        batch_size=64,
    ),
    "densenet121_finetune": ExperimentSpec(
        model_name="densenet121",
        init_mode="finetune",
        epochs=30,
        batch_size=64,
    ),
    "resnext50_32x4d_scratch": ExperimentSpec(
        model_name="resnext50_32x4d",
        init_mode="scratch",
        epochs=30,
        batch_size=48,
    ),
    "resnext50_32x4d_finetune": ExperimentSpec(
        model_name="resnext50_32x4d",
        init_mode="finetune",
        epochs=30,
        batch_size=48,
    ),
}

PRESET_ORDER: List[str] = [
    "densenet121_scratch",
    "densenet121_finetune",
    "resnext50_32x4d_scratch",
    "resnext50_32x4d_finetune",
]


def canonical_specs() -> List[ExperimentSpec]:
    return [CANONICAL_PRESETS[key] for key in PRESET_ORDER]


def get_spec(key: str) -> ExperimentSpec:
    try:
        return CANONICAL_PRESETS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown preset key: {key}") from exc
