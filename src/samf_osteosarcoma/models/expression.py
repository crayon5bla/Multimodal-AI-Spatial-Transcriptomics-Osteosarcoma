from torch import Tensor, nn
from torch.nn import functional as F


class ExpressionPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        genes: int = 2000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, genes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def cosine_expression_loss(predicted: Tensor, target: Tensor) -> Tensor:
    return (1 - F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)).mean()


def distillation_loss(student: Tensor, teacher: Tensor, temperature: float = 4.0) -> Tensor:
    scaled_student = F.log_softmax(student / temperature, dim=-1)
    scaled_teacher = F.softmax(teacher / temperature, dim=-1)
    return F.kl_div(scaled_student, scaled_teacher, reduction="batchmean") * temperature**2


def combined_expression_loss(
    student: Tensor,
    target: Tensor,
    teacher: Tensor,
    alpha: float = 0.3,
    temperature: float = 4.0,
) -> Tensor:
    direct = cosine_expression_loss(student, target)
    transferred = distillation_loss(student, teacher, temperature)
    return (1 - alpha) * direct + alpha * transferred
