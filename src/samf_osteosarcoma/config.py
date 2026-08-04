from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ExpressionSettings:
    hidden_dim: int = 512
    dropout: float = 0.1
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 100


@dataclass(frozen=True)
class DistillationSettings:
    alpha: float = 0.3
    temperature: float = 4.0


@dataclass(frozen=True)
class FusionSettings:
    pathway_count: int = 331
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 8
    dropout: float = 0.1


@dataclass(frozen=True)
class SurvivalSettings:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    l2: float = 1e-4
    epochs: int = 50
    patience: int = 15


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int = 42
    genes: int = 2000
    feature_dim: int = 1024
    expression: ExpressionSettings = field(default_factory=ExpressionSettings)
    distillation: DistillationSettings = field(default_factory=DistillationSettings)
    fusion: FusionSettings = field(default_factory=FusionSettings)
    survival: SurvivalSettings = field(default_factory=SurvivalSettings)


def load_mapping(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise TypeError("configuration root must be a mapping")
    return cast(dict[str, Any], loaded)
