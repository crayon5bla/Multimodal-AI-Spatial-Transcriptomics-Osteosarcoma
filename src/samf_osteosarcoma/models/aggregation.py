import torch
from torch import Tensor, nn


class GatedAttentionPool(nn.Module):
    def __init__(self, input_dim: int = 256, attention_dim: int = 128) -> None:
        super().__init__()
        self.tanh_branch = nn.Linear(input_dim, attention_dim)
        self.sigmoid_branch = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, instances: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        gated = torch.tanh(self.tanh_branch(instances))
        gated = gated * torch.sigmoid(self.sigmoid_branch(instances))
        logits = self.score(gated).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", weights, instances)
        return pooled, weights

