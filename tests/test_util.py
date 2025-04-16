import pytest
import numpy as np
from genlm.eval.util import bootstrap_ci


def test_bootstrap_ci_basic():
    # Test with simple mean calculation
    values = [1, 2, 3, 4, 5]
    mean, lower, upper = bootstrap_ci(values, np.mean, ci=0.95, n_bootstrap=1000)

    assert isinstance(mean, float)
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower <= mean <= upper
    assert abs(mean - 3.0) < 0.1  # Mean should be close to 3


def test_bootstrap_ci_different_metrics():
    values = [1, 2, 3, 4, 5]

    # Test with median
    mean, _, _ = bootstrap_ci(values, np.median, ci=0.95, n_bootstrap=1000)
    assert abs(mean - 3.0) < 0.1  # Median should be close to 3

    # Test with standard deviation
    mean, _, _ = bootstrap_ci(values, np.std, ci=0.95, n_bootstrap=1000)
    assert 1.0 < mean < 2.0  # std should be around 1.58


def test_bootstrap_ci_invalid_ci():
    values = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="ci must be between 0 and 1"):
        bootstrap_ci(values, np.mean, ci=1.5)


def test_bootstrap_ci_different_confidence_levels():
    values = [1, 2, 3, 4, 5]

    # Test with 90% confidence interval
    mean90, lower90, upper90 = bootstrap_ci(values, np.mean, ci=0.90, n_bootstrap=1000)

    # Test with 99% confidence interval
    mean99, lower99, upper99 = bootstrap_ci(values, np.mean, ci=0.99, n_bootstrap=1000)

    # 99% CI should be wider than 90% CI
    assert (upper99 - lower99) > (upper90 - lower90)
    # Means should be approximately equal
    assert abs(mean90 - mean99) < 0.1


def test_bootstrap_ci_empty_array():
    with pytest.raises(
        Exception
    ):  # Either ValueError or IndexError depending on metric
        bootstrap_ci([], np.mean)


def test_bootstrap_ci_single_value():
    values = [42]
    mean, lower, upper = bootstrap_ci(values, np.mean, ci=0.95, n_bootstrap=1000)
    assert mean == 42
    assert lower == 42
    assert upper == 42
