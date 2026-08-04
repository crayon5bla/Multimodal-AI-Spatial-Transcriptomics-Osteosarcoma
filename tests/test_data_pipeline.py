import numpy as np
import torch

from samf_osteosarcoma.data.expression import ExpressionTransform
from samf_osteosarcoma.data.tiles import iter_coordinates, tile_is_tissue


def test_expression_filter_and_normalize() -> None:
    counts = torch.ones(4, 2100) * 2
    transform = ExpressionTransform()
    matrix, indices = transform.fit_transform(counts)
    assert matrix.shape == (4, 2000)
    assert indices.shape == (2000,)


def test_tile_grid_nonoverlap() -> None:
    values = list(iter_coordinates(512, 512))
    assert [(value.x, value.y) for value in values] == [(0, 0), (256, 0), (0, 256), (256, 256)]


def test_blank_tile_rejected() -> None:
    assert not tile_is_tissue(np.full((256, 256, 3), 255, dtype=np.uint8))

