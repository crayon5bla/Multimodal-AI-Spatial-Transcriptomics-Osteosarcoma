import torch

from samf_osteosarcoma.models.expression import (
    ExpressionPredictor,
    combined_expression_loss,
    cosine_expression_loss,
)


def test_expression_shape() -> None:
    model = ExpressionPredictor(input_dim=16, hidden_dim=12, genes=20, dropout=0)
    output = model(torch.randn(7, 16))
    assert output.shape == (7, 20)


def test_cosine_identity() -> None:
    target = torch.randn(8, 20)
    assert torch.allclose(cosine_expression_loss(target, target), torch.tensor(0.0), atol=1e-6)


def test_combined_loss_has_gradient() -> None:
    student = torch.randn(8, 20, requires_grad=True)
    target = torch.rand(8, 20)
    teacher = torch.randn(8, 20)
    loss = combined_expression_loss(student, target, teacher)
    loss.backward()
    assert student.grad is not None

