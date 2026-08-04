from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ranksums
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class InterfaceResult:
    indices: np.ndarray
    distances: np.ndarray


def find_interface_tiles(
    viable_centroids: np.ndarray,
    necrotic_centroids: np.ndarray,
    distance_um: float = 200.0,
) -> InterfaceResult:
    if viable_centroids.ndim != 2 or viable_centroids.shape[1] != 2:
        raise ValueError("viable centroids must have x and y columns")
    if necrotic_centroids.ndim != 2 or necrotic_centroids.shape[1] != 2:
        raise ValueError("necrotic centroids must have x and y columns")
    tree = cKDTree(necrotic_centroids)
    distances, _ = tree.query(viable_centroids, k=1)
    indices = np.flatnonzero(distances <= distance_um)
    return InterfaceResult(indices=indices, distances=distances[indices])


def select_consensus_k(
    signatures: np.ndarray,
    candidates: tuple[int, ...] = (2, 3, 4, 5),
    seed: int = 42,
) -> tuple[int, np.ndarray]:
    selected_k = candidates[0]
    selected_labels = np.zeros(signatures.shape[0], dtype=np.int64)
    selected_score = -np.inf
    for clusters in candidates:
        labels = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit_predict(signatures)
        score = silhouette_score(signatures, labels)
        if score > selected_score:
            selected_k = clusters
            selected_labels = labels
            selected_score = score
    return selected_k, selected_labels


def differential_pathways(
    signatures: np.ndarray,
    labels: np.ndarray,
    positive_cluster: int,
) -> tuple[np.ndarray, np.ndarray]:
    first = signatures[labels == positive_cluster]
    second = signatures[labels != positive_cluster]
    statistics = np.empty(signatures.shape[1])
    probabilities = np.empty(signatures.shape[1])
    for pathway in range(signatures.shape[1]):
        result = ranksums(first[:, pathway], second[:, pathway])
        statistics[pathway] = result.statistic
        probabilities[pathway] = result.pvalue
    return statistics, probabilities

