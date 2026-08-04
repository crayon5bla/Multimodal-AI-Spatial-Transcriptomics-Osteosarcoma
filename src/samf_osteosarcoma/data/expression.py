from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ExpressionTransform:
    minimum_genes: int = 500
    minimum_umis: int = 1000
    output_genes: int = 2000

    def valid_spots(self, counts: Tensor) -> Tensor:
        if counts.ndim != 2:
            raise ValueError("counts must have shape spots by genes")
        detected = (counts > 0).sum(dim=1)
        totals = counts.sum(dim=1)
        return (detected >= self.minimum_genes) & (totals >= self.minimum_umis)

    def normalize(self, counts: Tensor) -> Tensor:
        totals = counts.sum(dim=1, keepdim=True).clamp_min(1)
        scaled = counts / totals * torch.median(totals)
        return torch.log2(scaled + 1)

    def select_hvgs(self, matrix: Tensor) -> Tensor:
        if matrix.ndim != 2:
            raise ValueError("matrix must have shape spots by genes")
        means = matrix.mean(dim=0)
        variances = matrix.var(dim=0, unbiased=False)
        dispersion = variances / means.clamp_min(1e-8)
        order = torch.argsort(dispersion, descending=True)
        return order[: self.output_genes]

    def fit_transform(self, counts: Tensor) -> tuple[Tensor, Tensor]:
        retained = self.valid_spots(counts)
        normalized = self.normalize(counts[retained])
        indices = self.select_hvgs(normalized)
        return normalized[:, indices], indices


def library_log2_numpy(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    safe = np.maximum(totals, 1)
    median = np.median(safe)
    return np.log2(counts / safe * median + 1)

