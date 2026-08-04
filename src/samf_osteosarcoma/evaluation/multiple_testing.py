import numpy as np


def benjamini_hochberg(probabilities: np.ndarray) -> np.ndarray:
    count = probabilities.size
    order = np.argsort(probabilities)
    ranked = probabilities[order]
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def holm_bonferroni(probabilities: np.ndarray) -> np.ndarray:
    count = probabilities.size
    order = np.argsort(probabilities)
    ranked = probabilities[order]
    adjusted = np.maximum.accumulate(ranked * np.arange(count, 0, -1))
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output

