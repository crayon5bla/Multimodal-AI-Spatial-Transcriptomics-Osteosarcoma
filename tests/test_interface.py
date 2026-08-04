import numpy as np

from samf_osteosarcoma.evaluation.interface import find_interface_tiles, select_consensus_k


def test_interface_distance() -> None:
    viable = np.array([[0.0, 0.0], [300.0, 0.0]])
    necrotic = np.array([[100.0, 0.0]])
    result = find_interface_tiles(viable, necrotic, 200)
    assert result.indices.tolist() == [0, 1]


def test_consensus_cluster_selection() -> None:
    generator = np.random.default_rng(4)
    first = generator.normal(-3, 0.1, size=(12, 3))
    second = generator.normal(3, 0.1, size=(12, 3))
    selected, labels = select_consensus_k(np.concatenate([first, second]), (2, 3))
    assert selected == 2
    assert np.unique(labels).size == 2

