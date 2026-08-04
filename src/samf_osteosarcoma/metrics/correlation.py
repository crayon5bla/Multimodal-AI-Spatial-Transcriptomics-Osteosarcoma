import numpy as np
from scipy.stats import pearsonr, spearmanr


def per_gene_pearson(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("arrays must have equal spots by genes shape")
    values = np.empty(predicted.shape[1], dtype=np.float64)
    for gene in range(predicted.shape[1]):
        values[gene] = pearsonr(predicted[:, gene], observed[:, gene]).statistic
    return values


def per_gene_spearman(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("arrays must have equal spots by genes shape")
    values = np.empty(predicted.shape[1], dtype=np.float64)
    for gene in range(predicted.shape[1]):
        values[gene] = spearmanr(predicted[:, gene], observed[:, gene]).statistic
    return values


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())

