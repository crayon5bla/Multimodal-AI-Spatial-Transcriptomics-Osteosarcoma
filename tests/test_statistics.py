import numpy as np

from samf_osteosarcoma.evaluation.bootstrap import bootstrap_interval
from samf_osteosarcoma.evaluation.multiple_testing import (
    benjamini_hochberg,
    holm_bonferroni,
)


def test_bootstrap_mean_interval() -> None:
    values = np.arange(100, dtype=np.float64)
    low, high = bootstrap_interval((values,), lambda x: float(x.mean()), iterations=200)
    assert low < values.mean() < high


def test_adjustments_are_bounded() -> None:
    probabilities = np.array([0.001, 0.02, 0.5])
    for adjusted in (benjamini_hochberg(probabilities), holm_bonferroni(probabilities)):
        assert np.all(adjusted >= probabilities)
        assert np.all(adjusted <= 1)

