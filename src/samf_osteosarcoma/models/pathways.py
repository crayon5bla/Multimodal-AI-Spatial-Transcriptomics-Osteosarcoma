from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PathwayIndex:
    names: tuple[str, ...]
    membership: Tensor

    def validate(self, genes: int = 2000, pathways: int = 331) -> None:
        if self.membership.shape != (pathways, genes):
            raise ValueError("pathway membership shape mismatch")
        if len(self.names) != pathways:
            raise ValueError("pathway name count mismatch")
        coverage = self.membership.sum(dim=1)
        if torch.any(coverage == 0):
            raise ValueError("empty pathways are not allowed")


class PathwayTokenizer(nn.Module):
    def __init__(self, membership: Tensor, hidden_dim: int = 256) -> None:
        super().__init__()
        if membership.ndim != 2:
            raise ValueError("membership must have shape pathways by genes")
        normalized = membership.float() / membership.sum(dim=1, keepdim=True).clamp_min(1)
        self.register_buffer("membership", normalized)
        self.gene_embeddings = nn.Parameter(torch.empty(membership.shape[1], hidden_dim))
        nn.init.trunc_normal_(self.gene_embeddings, std=0.02)

    def forward(self, expression: Tensor) -> Tensor:
        weighted = expression.unsqueeze(-1) * self.gene_embeddings
        return torch.einsum("kg,bgd->bkd", self.membership, weighted)

