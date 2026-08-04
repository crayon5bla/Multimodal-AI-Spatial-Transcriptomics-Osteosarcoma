import torch
from torch import Tensor, nn


def cox_partial_likelihood(risk: Tensor, duration: Tensor, event: Tensor) -> Tensor:
    if risk.ndim != 1 or duration.ndim != 1 or event.ndim != 1:
        raise ValueError("survival tensors must be one-dimensional")
    order = torch.argsort(duration, descending=True)
    ordered_risk = risk[order]
    ordered_event = event[order].float()
    log_cumulative = torch.logcumsumexp(ordered_risk, dim=0)
    contributions = (ordered_risk - log_cumulative) * ordered_event
    return -contributions.sum() / ordered_event.sum().clamp_min(1)


def l2_penalty(module: nn.Module) -> Tensor:
    parameters = [parameter.square().sum() for parameter in module.parameters()]
    if not parameters:
        return torch.zeros(())
    return torch.stack(parameters).sum()


def survival_objective(
    module: nn.Module,
    risk: Tensor,
    duration: Tensor,
    event: Tensor,
    coefficient: float = 1e-4,
) -> Tensor:
    return cox_partial_likelihood(risk, duration, event) + coefficient * l2_penalty(module)

