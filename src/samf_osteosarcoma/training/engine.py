from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from samf_osteosarcoma.losses.survival import survival_objective


class SurvivalBatch(Protocol):
    features: Tensor
    mask: Tensor
    duration: Tensor
    event: Tensor


@dataclass
class EpochSummary:
    loss: float
    examples: int


def train_survival_epoch(
    model: nn.Module,
    batches: list[SurvivalBatch],
    optimizer: Optimizer,
    device: torch.device,
    l2: float = 1e-4,
) -> EpochSummary:
    model.train()
    total = 0.0
    examples = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        features = batch.features.to(device)
        mask = batch.mask.to(device)
        duration = batch.duration.to(device)
        event = batch.event.to(device)
        output = model(features, mask)
        loss = survival_objective(model, output["risk"], duration, event, l2)
        loss.backward()
        optimizer.step()
        count = int(duration.shape[0])
        total += float(loss.detach()) * count
        examples += count
    return EpochSummary(loss=total / max(examples, 1), examples=examples)

