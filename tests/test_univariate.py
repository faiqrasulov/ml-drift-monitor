import math

import numpy as np
import pytest
from scipy import special, stats
from scipy.spatial.distance import jensenshannon
from statsmodels.stats.proportion import proportions_ztest

from driftwatch.univariate import (
    chi2_homogeneity,
    js_distance,
    js_distance_categorical,
    kolmogorov_sf,
    ks_statistic,
    ks_test,
    missing_rate_test,
    normalized_wasserstein,
    psi,
    psi_categorical,
    psi_from_proportions,
    quantile_bin_edges,
    wasserstein_distance,
)


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) == 0.0
    assert psi(x, x.copy()[::-1], n_bins=20) == 0.0
    assert psi_categorical(["a", "b", "b"], ["b", "a", "b"]) == 0.0


def test_psi_matches_hand_computed_case():
    # Reference 1..100 -> quartile edges 25.75, 50.5, 75.25 -> 25% in each bin.
    ref = np.arange(1, 101, dtype=float)
    np.testing.assert_allclose(quantile_bin_edges(ref, 4), [25.75, 50.5, 75.25])
    cur = np.repeat([1.0, 30.0, 60.0, 90.0], [10, 20, 30, 40])  # 10/20/30/40%
    by_hand = (
        (0.1 - 0.25) * math.log(0.1 / 0.25)
        + (0.2 - 0.25) * math.log(0.2 / 0.25)
        + (0.3 - 0.25) * math.log(0.3 / 0.25)
        + (0.4 - 0.25) * math.log(0.4 / 0.25)
    )
    assert psi(ref, cur, n_bins=4) == pytest.approx(by_hand, abs=1e-12)
    assert by_hand == pytest.approx(0.228218, abs=1e-6)


def test_psi_floors_empty_bins_at_eps():
    ref = np.arange(1, 101, dtype=float)
    cur = np.ones(50)  # everything in the first bin, three empty bins
    eps = 1e-4
    expected = (1 - 0.25) * math.log(1 / 0.25) + 3 * (eps - 0.25) * math.log(eps / 0.25)
    value = psi(ref, cur, n_bins=4, eps=eps)
    assert math.isfinite(value)
    assert value == pytest.approx(expected, rel=1e-12)
    assert math.isfinite(psi_from_proportions([0.5, 0.5], [1.0, 0.0], eps=1e-6))


def test_psi_is_symmetric_in_proportions_and_ignores_missing():
    p, q = np.array([0.2, 0.3, 0.5]), np.array([0.4, 0.4, 0.2])
    assert psi_from_proportions(p, q) == pytest.approx(psi_from_proportions(q, p))
    x = np.random.default_rng(1).normal(size=500)
    with_nan = np.concatenate([x, [np.nan] * 50])
    assert psi(x, with_nan) == 0.0


def test_bin_edges_handle_ties_and_outliers():
    ref = np.repeat([1.0, 2.0, 3.0], [70, 20, 10])
    edges = quantile_bin_edges(ref, 10)
    assert len(edges) < 9  # duplicate quantiles collapsed
    assert np.all(np.diff(edges) > 0)
    # values far outside the reference range still land in the open outer bins
    assert math.isfinite(psi(ref, [-100.0, 100.0, 2.0]))


@pytest.mark.parametrize("seed", range(6))
def test_ks_statistic_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    a = np.round(rng.normal(size=rng.integers(20, 400)), int(seed % 3))  # ties for some seeds
    b = np.round(rng.normal(0.2, 1.2, size=rng.integers(20, 400)), int(seed % 3))
    ref = stats.ks_2samp(a, b, method="asymp")
    exact = stats.ks_2samp(a, b, method="exact")
    assert ks_statistic(a, b) == pytest.approx(ref.statistic, abs=1e-12)
    # Stephens-corrected asymptotic p-value: within 0.02 of scipy everywhere and within 5%
    # (relative) where it matters, in the tail.
    p = ks_test(a, b).p_value
    assert p == pytest.approx(ref.pvalue, abs=0.02)
    assert p == pytest.approx(exact.pvalue, abs=0.02)
    if ref.pvalue < 0.2:
        assert p == pytest.approx(ref.pvalue, rel=0.05)


def test_kolmogorov_sf_matches_scipy():
    for x in [0.25, 0.5, 0.8, 1.0, 1.36, 2.0, 3.0]:
        assert kolmogorov_sf(x) == pytest.approx(special.kolmogorov(x), abs=1e-10)


@pytest.mark.parametrize("seed", range(4))
def test_wasserstein_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    u, v = rng.exponential(size=300), rng.exponential(1.5, size=170)
    assert wasserstein_distance(u, v) == pytest.approx(stats.wasserstein_distance(u, v), 1e-10)


def test_normalized_wasserstein_is_shift_in_sd_units():
    x = np.random.default_rng(3).normal(10, 4, 5000)
    assert normalized_wasserstein(x, x + 2.0) == pytest.approx(2.0 / x.std(), rel=1e-10)
    assert normalized_wasserstein(x, x + 2.0) == pytest.approx(0.5, abs=0.01)


def test_js_distance_bounds_and_scipy():
    assert js_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0, abs=1e-12)
    assert js_distance([1, 0], [0, 1]) == pytest.approx(1.0)  # disjoint supports
    rng = np.random.default_rng(4)
    for _ in range(20):
        p, q = rng.random(6), rng.random(6)
        d = js_distance(p, q)
        assert 0.0 <= d <= 1.0
        assert d == pytest.approx(jensenshannon(p, q, base=2), abs=1e-12)
    assert js_distance_categorical(["a", "b"], ["c", "d"]) == pytest.approx(1.0)


def test_chi2_homogeneity_matches_scipy():
    rng = np.random.default_rng(5)
    ref = rng.choice(list("abcd"), 600, p=[0.4, 0.3, 0.2, 0.1])
    cur = rng.choice(list("abcde"), 300, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    res = chi2_homogeneity(ref, cur)
    cats = sorted(set(ref) | set(cur))
    table = [[np.sum(ref == c) for c in cats], [np.sum(cur == c) for c in cats]]
    stat, p, dof, _ = stats.chi2_contingency(table, correction=False)
    assert res.statistic == pytest.approx(stat, rel=1e-12)
    assert res.p_value == pytest.approx(p, rel=1e-9)
    assert res.dof == dof


def test_missing_rate_test_matches_statsmodels():
    ref = np.where(np.arange(1000) < 20, np.nan, 1.0)  # 2% missing
    cur = np.where(np.arange(400) < 30, np.nan, 1.0)  # 7.5% missing
    res = missing_rate_test(ref, cur)
    z, p = proportions_ztest([30, 20], [400, 1000])
    assert res.z == pytest.approx(z, rel=1e-10)
    assert res.p_value == pytest.approx(p, rel=1e-10)
    assert missing_rate_test([1.0, 2.0], [3.0]).p_value == 1.0
