import torch

from samf_osteosarcoma.models.samf import SAMF


def test_end_to_end_forward_backward() -> None:
    membership = torch.zeros(5, 20)
    for index in range(5):
        membership[index, index * 4 : (index + 1) * 4] = 1
    model = SAMF(
        membership=membership,
        feature_dim=16,
        genes=20,
        hidden_dim=8,
        layers=2,
        heads=2,
    )
    output = model(torch.randn(3, 7, 16))
    assert output["risk"].shape == (3,)
    assert output["expression"].shape == (3, 7, 20)
    output["risk"].sum().backward()
    assert model.risk_head.weight.grad is not None

