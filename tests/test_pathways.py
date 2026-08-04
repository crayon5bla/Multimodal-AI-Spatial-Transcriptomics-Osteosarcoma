import torch

from samf_osteosarcoma.models.pathways import PathwayIndex, PathwayTokenizer


def test_pathway_index_validation() -> None:
    membership = torch.ones(3, 5)
    index = PathwayIndex(("a", "b", "c"), membership)
    index.validate(genes=5, pathways=3)


def test_tokenizer_shape() -> None:
    membership = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    tokenizer = PathwayTokenizer(membership, hidden_dim=6)
    tokens = tokenizer(torch.rand(3, 4))
    assert tokens.shape == (3, 2, 6)

