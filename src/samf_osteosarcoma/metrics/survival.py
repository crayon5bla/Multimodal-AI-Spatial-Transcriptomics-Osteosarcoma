import numpy as np


def concordance_index(duration: np.ndarray, risk: np.ndarray, event: np.ndarray) -> float:
    if not (duration.shape == risk.shape == event.shape):
        raise ValueError("survival arrays must have equal shape")
    concordant = 0.0
    comparable = 0
    for left in range(duration.size):
        for right in range(left + 1, duration.size):
            if duration[left] == duration[right]:
                continue
            early, late = (left, right) if duration[left] < duration[right] else (right, left)
            if not bool(event[early]):
                continue
            comparable += 1
            if risk[early] > risk[late]:
                concordant += 1
            elif risk[early] == risk[late]:
                concordant += 0.5
    if comparable == 0:
        raise ValueError("no comparable survival pairs")
    return concordant / comparable

