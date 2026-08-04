from torch import Tensor, nn


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int = 256, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.pathway_to_histology = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.histology_to_pathway = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pathway_norm = nn.LayerNorm(hidden_dim)
        self.histology_norm = nn.LayerNorm(hidden_dim)
        self.pathway_feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.histology_feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, pathways: Tensor, histology: Tensor) -> tuple[Tensor, Tensor]:
        pathway_update, _ = self.pathway_to_histology(pathways, histology, histology)
        histology_update, _ = self.histology_to_pathway(histology, pathways, pathways)
        pathways = self.pathway_norm(pathways + pathway_update)
        histology = self.histology_norm(histology + histology_update)
        pathways = self.pathway_norm(pathways + self.pathway_feedforward(pathways))
        histology = self.histology_norm(histology + self.histology_feedforward(histology))
        return pathways, histology


class CrossAttentionStack(nn.Module):
    def __init__(
        self,
        layers: int = 4,
        hidden_dim: int = 256,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            BidirectionalCrossAttention(hidden_dim, heads, dropout) for _ in range(layers)
        )

    def forward(self, pathways: Tensor, histology: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            pathways, histology = layer(pathways, histology)
        return pathways, histology
