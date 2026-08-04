from collections.abc import Callable

import numpy as np


def bootstrap_interval(
    arrays: tuple[np.ndarray, ...],
    statistic: Callable[..., float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if not arrays:
        raise ValueError("at least one array is required")
    size = arrays[0].shape[0]
    if any(array.shape[0] != size for array in arrays):
        raise ValueError("bootstrap arrays must align")
    generator = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = generator.integers(0, size, size=size)
        estimates[index] = statistic(*(array[sample] for array in arrays))
    tail = (1 - confidence) / 2
    low, high = np.quantile(estimates, [tail, 1 - tail])
    return float(low), float(high)


def permutation_difference(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    iterations: int = 1000,
    seed: int = 42,
) -> float:
    observed = statistic(labels, first) - statistic(labels, second)
    generator = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = generator.permutation(labels)
        difference = statistic(permuted, first) - statistic(permuted, second)
        extreme += int(difference >= observed)
    return (extreme + 1) / (iterations + 1)

