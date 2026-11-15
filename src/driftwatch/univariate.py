"""Per-feature two-sample drift statistics, implemented from scratch with numpy.

scipy is used only for the chi-square tail probability (``scipy.special.chdtrc``); the test
statistics themselves are computed here and checked against scipy in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special

DEFAULT_EPS = 1e-4


@dataclass(frozen=True)
class TestResult:
    """A test statistic with its p-value (``None`` when the statistic has no p-value)."""

    statistic: float
    p_value: float | None = None
    dof: int | None = None


@dataclass(frozen=True)
class MissingRateResult:
    """Missing-value rates in both samples and a two-proportion z-test of their difference."""

    reference_rate: float
    current_rate: float
    z: float
    p_value: float


def finite_values(x: object) -> np.ndarray:
    """Return the non-missing values of a numeric array-like as float64."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[~np.isnan(arr)]


def non_missing(x: object) -> np.ndarray:
    """Return the non-missing values of any array-like (numeric or categorical) as an array."""
    s = pd.Series(np.asarray(x, dtype=object).ravel())
    return s[s.notna()].to_numpy(dtype=object)


# --------------------------------------------------------------------------- PSI


def quantile_bin_edges(reference: object, n_bins: int = 10) -> np.ndarray:
    """Interior bin edges at the reference deciles (or other quantiles).

    The outer bins are open (-inf, e1] and (e_k, inf), so values outside the reference range
    still land in a bin. Duplicate edges (from ties in discrete features) are removed, so the
    effective number of bins can be smaller than ``n_bins``.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    ref = finite_values(reference)
    if ref.size == 0:
        raise ValueError("reference has no non-missing values")
    qs = np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.unique(qs)


def bin_counts(x: object, edges: np.ndarray) -> np.ndarray:
    """Counts of ``x`` in the right-closed bins defined by interior ``edges``."""
    values = finite_values(x)
    idx = np.searchsorted(edges, values, side="left")
    return np.bincount(idx, minlength=len(edges) + 1)


def psi_from_proportions(
    p_reference: np.ndarray, p_current: np.ndarray, eps: float = DEFAULT_EPS
) -> float:
    """Population Stability Index sum((c - r) * ln(c / r)) over bins.

    Empty bins are floored at ``eps`` before taking logs, so one empty bin contributes about
    ``p * ln(p / eps)`` instead of infinity.
    """
    p = np.maximum(np.asarray(p_reference, dtype=np.float64), eps)
    q = np.maximum(np.asarray(p_current, dtype=np.float64), eps)
    return float(np.sum((q - p) * np.log(q / p)))


def psi(
    reference: object,
    current: object,
    n_bins: int = 10,
    eps: float = DEFAULT_EPS,
    edges: np.ndarray | None = None,
) -> float:
    """PSI of a numeric feature with bins at reference quantiles (missing values ignored)."""
    if edges is None:
        edges = quantile_bin_edges(reference, n_bins)
    r = bin_counts(reference, edges)
    c = bin_counts(current, edges)
    if r.sum() == 0 or c.sum() == 0:
        return float("nan")
    return psi_from_proportions(r / r.sum(), c / c.sum(), eps)


def category_proportions(
    reference: object, current: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Categories (union of both samples) and their proportions in each sample."""
    r = pd.Series(non_missing(reference)).astype(str)
    c = pd.Series(non_missing(current)).astype(str)
    cats = np.array(sorted(set(r.unique()) | set(c.unique())), dtype=object)
    rc = r.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=np.float64)
    cc = c.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=np.float64)
    if rc.sum() == 0 or cc.sum() == 0:
        return cats, rc, cc
    return cats, rc / rc.sum(), cc / cc.sum()


def psi_categorical(reference: object, current: object, eps: float = DEFAULT_EPS) -> float:
    """PSI of a categorical feature, one bin per category (unseen categories get ``eps``)."""
    _, p, q = category_proportions(reference, current)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    return psi_from_proportions(p, q, eps)


# --------------------------------------------------------------------------- KS


def ks_statistic_sorted(x: np.ndarray, y: np.ndarray) -> float:
    """KS statistic for two already sorted, NaN-free samples."""
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def ks_statistic(a: object, b: object) -> float:
    """Two-sample Kolmogorov-Smirnov statistic sup_x |F_a(x) - F_b(x)| (ties handled exactly)."""
    x = np.sort(finite_values(a))
    y = np.sort(finite_values(b))
    if x.size == 0 or y.size == 0:
        return float("nan")
    return ks_statistic_sorted(x, y)


def ks_p_value(d: float, n: int, m: int) -> float:
    """Asymptotic two-sided KS p-value with Stephens' small-sample correction."""
    en = math.sqrt(n * m / (n + m))
    return kolmogorov_sf((en + 0.12 + 0.11 / en) * d)


def kolmogorov_sf(lam: float) -> float:
    """P(K > lam) for the Kolmogorov distribution: 2 * sum_k (-1)^(k-1) exp(-2 k^2 lam^2)."""
    if lam < 0.2:
        return 1.0  # the alternating series is numerically 1 here and converges slowly
    total = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * k * k * lam * lam)
        total += term if k % 2 else -term
        if term < 1e-16:
            break
    return float(min(max(2.0 * total, 0.0), 1.0))


def ks_test(a: object, b: object) -> TestResult:
    """Two-sample KS test with the asymptotic p-value and Stephens' small-sample correction."""
    d = ks_statistic(a, b)
    if math.isnan(d):
        return TestResult(d, None)
    return TestResult(d, ks_p_value(d, finite_values(a).size, finite_values(b).size))


# --------------------------------------------------------------------------- Wasserstein


def wasserstein_distance(u: object, v: object) -> float:
    """1-Wasserstein (earth mover's) distance between two empirical distributions.

    Equal to the integral of |F_u(x) - F_v(x)| dx, evaluated exactly on the merged sample.
    """
    x = np.sort(finite_values(u))
    y = np.sort(finite_values(v))
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([x, y]))
    deltas = np.diff(grid)
    cdf_x = np.searchsorted(x, grid[:-1], side="right") / x.size
    cdf_y = np.searchsorted(y, grid[:-1], side="right") / y.size
    return float(np.sum(np.abs(cdf_x - cdf_y) * deltas))


def normalized_wasserstein(reference: object, current: object) -> float:
    """Wasserstein distance divided by the reference standard deviation (scale-free).

    0.1 means the current distribution is, on average, shifted by a tenth of a reference SD.
    Falls back to the raw distance when the reference is constant.
    """
    ref = finite_values(reference)
    w = wasserstein_distance(ref, current)
    sd = float(np.std(ref)) if ref.size else 0.0
    return w / sd if sd > 0 else w


# --------------------------------------------------------------------------- categorical


def js_distance(p: object, q: object, base: float = 2.0) -> float:
    """Jensen-Shannon distance sqrt(JSD(p, q)); in [0, 1] for base 2.

    Inputs are non-negative weights and are normalised to probability vectors.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if (p < 0).any() or (q < 0).any() or p.sum() == 0 or q.sum() == 0:
        raise ValueError("p and q must be non-negative with positive sum")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    jsd /= math.log(base)
    return math.sqrt(max(jsd, 0.0))


def js_distance_categorical(reference: object, current: object) -> float:
    """Jensen-Shannon distance (base 2) between category frequencies of two samples."""
    _, p, q = category_proportions(reference, current)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    return js_distance(p, q)


def chi2_homogeneity(reference: object, current: object) -> TestResult:
    """Chi-square test that two samples share the same category distribution (2 x K table)."""
    r = pd.Series(non_missing(reference)).astype(str).value_counts()
    c = pd.Series(non_missing(current)).astype(str).value_counts()
    cats = sorted(set(r.index) | set(c.index))
    table = np.vstack(
        [r.reindex(cats, fill_value=0).to_numpy(), c.reindex(cats, fill_value=0).to_numpy()]
    ).astype(np.float64)
    if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
        return TestResult(0.0, 1.0, 0)
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    stat = float(np.sum((table - expected) ** 2 / expected))
    dof = table.shape[1] - 1
    return TestResult(stat, float(special.chdtrc(dof, stat)), dof)


# --------------------------------------------------------------------------- missing values


def normal_sf(z: float) -> float:
    """Upper tail of the standard normal, 1 - Phi(z)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def missing_rate_test(reference: object, current: object) -> MissingRateResult:
    """Two-sided pooled two-proportion z-test for a change in the missing-value rate."""
    r = pd.isna(pd.Series(np.asarray(reference, dtype=object).ravel()))
    c = pd.isna(pd.Series(np.asarray(current, dtype=object).ravel()))
    n1, n2 = len(r), len(c)
    if n1 == 0 or n2 == 0:
        raise ValueError("both samples must be non-empty")
    p1, p2 = float(r.mean()), float(c.mean())
    pooled = (r.sum() + c.sum()) / (n1 + n2)
    if pooled in (0.0, 1.0):
        return MissingRateResult(p1, p2, 0.0, 1.0)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return MissingRateResult(p1, p2, z, min(1.0, 2.0 * normal_sf(abs(z))))
