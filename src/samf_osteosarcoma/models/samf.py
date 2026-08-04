import torch
from torch import Tensor, nn

from samf_osteosarcoma.models.aggregation import GatedAttentionPool
from samf_osteosarcoma.models.attention import CrossAttentionStack
from samf_osteosarcoma.models.expression import ExpressionPredictor
from samf_osteosarcoma.models.pathways import PathwayTokenizer


class SAMF(nn.Module):
    def __init__(
        self,
        membership: Tensor,
        feature_dim: int = 1024,
        genes: int = 2000,
        hidden_dim: int = 256,
        layers: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.expression = ExpressionPredictor(feature_dim, 512, genes, 0.1)
        self.tokenizer = PathwayTokenizer(membership, hidden_dim)
        self.histology_projection = nn.Linear(feature_dim, hidden_dim)
        self.fusion = CrossAttentionStack(layers, hidden_dim, heads, 0.1)
        self.pool = GatedAttentionPool(hidden_dim)
        self.risk_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, features: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        expression = self.expression(features)
        slide_expression = expression.mean(dim=1)
        pathways = self.tokenizer(slide_expression)
        histology = self.histology_projection(features)
        pathways, histology = self.fusion(pathways, histology)
        histology_summary, attention = self.pool(histology, mask)
        pathway_summary = pathways.mean(dim=1)
        fused = torch.cat([histology_summary, pathway_summary], dim=-1)
        risk = self.risk_head(fused).squeeze(-1)
        return {
            "risk": risk,
            "expression": expression,
            "pathways": pathways,
            "attention": attention,
        }

