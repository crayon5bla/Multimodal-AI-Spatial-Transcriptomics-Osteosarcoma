from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpotRecord:
    patient_id: str
    slide_id: str
    spot_id: str
    image_path: Path
    x: float
    y: float
    counts: Tensor

    def validate(self, genes: int = 2000) -> None:
        if self.counts.ndim != 1 or self.counts.shape[0] != genes:
            raise ValueError("spot expression dimension mismatch")
        if torch.any(self.counts < 0):
            raise ValueError("counts must be nonnegative")


@dataclass(frozen=True)
class SlideOutcome:
    patient_id: str
    slide_id: str
    duration: float
    event: bool
    metastasis: bool

    def validate(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")

