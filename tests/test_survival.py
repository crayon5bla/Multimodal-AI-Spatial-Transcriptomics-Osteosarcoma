import numpy as np
import torch

from samf_osteosarcoma.losses.survival import cox_partial_likelihood
from samf_osteosarcoma.metrics.survival import concordance_index


def test_concordance_perfect_order() -> None:
    duration = np.array([1.0, 2.0, 3.0])
    risk = np.array([3.0, 2.0, 1.0])
    event = np.array([1, 1, 1])
    assert concordance_index(duration, risk, event) == 1.0


def test_cox_loss_finite() -> None:
    risk = torch.tensor([0.3, -0.1, 0.8], requires_grad=True)
    loss = cox_partial_likelihood(risk, torch.tensor([2.0, 3.0, 1.0]), torch.ones(3))
    loss.backward()
    assert torch.isfinite(loss)

