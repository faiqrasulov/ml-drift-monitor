import numpy as np
import pytest

from driftwatch.calibration import (
    PSICalibrator,
    bootstrap_p_value,
    chi2_psi_p_value,
    chi2_psi_threshold,
    partition_quantiles,
    psi_against,
    psi_null_categorical,
    psi_null_distribution,
    psi_sorted,
    reference_bins,
)
from driftwatch.univariate import psi


def test_fast_paths_match_reference_implementation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ref = np.round(rng.lognormal(size=rng.integers(20, 2000)), int(rng.integers(0, 3)))
        cur = np.round(rng.lognormal(0.1, size=rng.integers(5, 500)), 1)
        b = int(rng.integers(2, 25))
        assert psi_sorted(np.sort(ref), np.sort(cur), b) == pytest.approx(psi(ref, cur, b), 1e-12)
        edges, p_ref = reference_bins(np.sort(ref), b)
        assert psi_against(edges, p_ref, np.sort(cur)) == pytest.approx(psi(ref, cur, b), 1e-12)
        probs = np.linspace(0, 1, b + 1)[1:-1]
        np.testing.assert_allclose(partition_quantiles(ref, probs), np.quantile(ref, probs))


def test_chi2_threshold_known_value():
    # chi2(9) 95th percentile = 16.919; times (1/1000 + 1/1000)
    assert chi2_psi_threshold(1000, 1000, 10, 0.05) == pytest.approx(16.919 * 0.002, rel=1e-4)
    t = chi2_psi_threshold(2000, 300, 10, 0.05)
    assert chi2_psi_p_value(t, 2000, 300, 10) == pytest.approx(0.05, rel=1e-9)


def test_bootstrap_null_mean_matches_chi_square_theory():
    ref = np.random.default_rng(1).normal(size=4000)
    null = psi_null_distribution(ref, n_current=1000, n_bins=10, n_boot=400, seed=2)
    theory = 9 * (1 / 4000 + 1 / 1000)  # E[PSI] ~ (B - 1)(1/n_ref + 1/n_cur)
    assert null.mean() == pytest.approx(theory, rel=0.15)


def test_bootstrap_p_value_bounds():
    null = np.linspace(0, 1, 99)
    assert bootstrap_p_value(null, 10.0) == pytest.approx(1 / 100)
    assert bootstrap_p_value(null, -1.0) == 1.0


def test_calibrated_threshold_controls_false_alarms_where_rule_of_thumb_fails():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=2000)
    cal = PSICalibrator(ref, n_bins=20, n_boot=500, seed=4)
    n = 50
    t = cal.threshold(n, alpha=0.05)
    fresh = [psi(ref, rng.normal(size=n), n_bins=20) for _ in range(600)]
    far_calibrated = np.mean(np.array(fresh) > t)
    far_rule = np.mean(np.array(fresh) > 0.2)
    assert 0.01 <= far_calibrated <= 0.10
    assert far_rule > 0.5  # "PSI > 0.2" with 20 bins and 50 rows alarms most of the time
    assert cal.threshold(1000) < cal.threshold(100) < t  # threshold shrinks with window size
    assert len(cal._cache) == 3  # one bootstrap per window size


def test_categorical_null_is_small_for_large_samples():
    ref = np.random.default_rng(5).choice(list("abc"), 3000, p=[0.6, 0.3, 0.1])
    null = psi_null_categorical(ref, 1000, n_boot=200, seed=6)
    assert null.mean() == pytest.approx(2 * (1 / 3000 + 1 / 1000), rel=0.3)
    with pytest.raises(ValueError):
        PSICalibrator(ref, kind="text")
