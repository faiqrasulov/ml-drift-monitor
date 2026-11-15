"""Calibrated PSI thresholds: bootstrap null distributions and a chi-square approximation.

Under no drift, the PSI between two samples of sizes n_ref and n_cur is approximately
(1/n_ref + 1/n_cur) * chi2(B - 1) for B bins, so a fixed cut-off such as 0.2 has a false-alarm
rate that depends on the sample sizes and the bin count. Calibrating the threshold on the
reference data fixes the false-alarm rate instead of the cut-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special

from driftwatch.univariate import DEFAULT_EPS, finite_values, non_missing


def _psi_counts(r_counts: np.ndarray, c_counts: np.ndarray, eps: float) -> float:
    p = np.maximum(r_counts / r_counts.sum(), eps)
    q = np.maximum(c_counts / c_counts.sum(), eps)
    return float(np.sum((q - p) * np.log(q / p)))


def sorted_quantiles(s: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Quantiles of a sorted array with numpy's default linear interpolation."""
    pos = probs * (s.size - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, s.size - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def partition_quantiles(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Same values as ``np.quantile(x, probs)`` (linear), via one partial sort (O(n))."""
    pos = probs * (x.size - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, x.size - 1)
    part = np.partition(x, np.unique(np.concatenate([lo, hi])))
    return part[lo] + (pos - lo) * (part[hi] - part[lo])


def sorted_bin_counts(s: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts of a sorted array in the right-closed bins defined by interior ``edges``."""
    cum = np.searchsorted(s, edges, side="right")
    return np.diff(np.concatenate(([0], cum, [s.size])))


def reference_bins(
    ref_sorted: np.ndarray, n_bins: int, eps: float = DEFAULT_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile bin edges of a sorted reference and its eps-floored bin proportions."""
    edges = np.unique(sorted_quantiles(ref_sorted, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    return edges, np.maximum(sorted_bin_counts(ref_sorted, edges) / ref_sorted.size, eps)


def psi_against(
    edges: np.ndarray, p_ref: np.ndarray, cur_sorted: np.ndarray, eps: float = DEFAULT_EPS
) -> float:
    """PSI of a sorted current sample against precomputed reference bins."""
    q = np.maximum(sorted_bin_counts(cur_sorted, edges) / cur_sorted.size, eps)
    return float(np.sum((q - p_ref) * np.log(q / p_ref)))


def psi_sorted(
    ref_sorted: np.ndarray, cur_sorted: np.ndarray, n_bins: int, eps: float = DEFAULT_EPS
) -> float:
    """PSI for sorted, NaN-free samples; same result as ``psi`` but reusable bins (fast path)."""
    edges, p_ref = reference_bins(ref_sorted, n_bins, eps)
    return psi_against(edges, p_ref, cur_sorted, eps)


def psi_null_distribution(
    reference: object,
    n_current: int,
    n_bins: int = 10,
    n_boot: int = 1000,
    eps: float = DEFAULT_EPS,
    seed: int = 0,
    n_reference: int | None = None,
) -> np.ndarray:
    """Bootstrap distribution of numeric PSI when there is no drift.

    Each replicate draws a pseudo-reference (size ``n_reference``, default: the reference size)
    and a pseudo-current window (size ``n_current``) with replacement from the reference,
    recomputes the quantile bins on the pseudo-reference, and records the PSI. This mirrors how
    the statistic is computed in production, including bin re-estimation, ties and the eps floor.
    """
    ref = finite_values(reference)
    if ref.size < 2:
        raise ValueError("reference needs at least 2 non-missing values")
    n_ref = int(n_reference or ref.size)
    probs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        r = ref[rng.integers(0, ref.size, n_ref)]
        c = ref[rng.integers(0, ref.size, int(n_current))]
        edges = np.unique(partition_quantiles(r, probs))
        k = edges.size + 1
        rc = np.bincount(np.searchsorted(edges, r, side="left"), minlength=k)
        cc = np.bincount(np.searchsorted(edges, c, side="left"), minlength=k)
        out[b] = _psi_counts(rc, cc, eps)
    return out


def psi_null_categorical(
    reference: object,
    n_current: int,
    n_boot: int = 1000,
    eps: float = DEFAULT_EPS,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap distribution of categorical PSI (one bin per category) under no drift."""
    values = pd.Series(non_missing(reference)).astype(str)
    if values.size < 2:
        raise ValueError("reference needs at least 2 non-missing values")
    codes = pd.factorize(values)[0]
    k = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        rc = np.bincount(codes[rng.integers(0, codes.size, codes.size)], minlength=k)
        cc = np.bincount(codes[rng.integers(0, codes.size, int(n_current))], minlength=k)
        out[b] = _psi_counts(rc, cc, eps)
    return out


def bootstrap_p_value(null: np.ndarray, observed: float) -> float:
    """Upper-tail bootstrap p-value (1 + #{null >= observed}) / (1 + B); never exactly 0."""
    null = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(null >= observed)) / (1 + null.size))


def chi2_psi_threshold(n_reference: int, n_current: int, n_bins: int, alpha: float = 0.05) -> float:
    """PSI threshold with false-alarm rate ``alpha`` under the chi-square approximation."""
    return float(special.chdtri(n_bins - 1, alpha) * (1.0 / n_reference + 1.0 / n_current))


def chi2_psi_p_value(value: float, n_reference: int, n_current: int, n_bins: int) -> float:
    """P(PSI >= value | no drift) under the chi-square approximation."""
    scale = 1.0 / n_reference + 1.0 / n_current
    return float(special.chdtrc(n_bins - 1, value / scale))


class PSICalibrator:
    """Bootstrap null distributions of PSI for one reference column, cached per window size."""

    def __init__(
        self,
        reference: object,
        kind: str = "numeric",
        n_bins: int = 10,
        n_boot: int = 1000,
        eps: float = DEFAULT_EPS,
        seed: int = 0,
    ) -> None:
        if kind not in ("numeric", "categorical"):
            raise ValueError("kind must be 'numeric' or 'categorical'")
        self.reference = reference
        self.kind = kind
        self.n_bins = n_bins
        self.n_boot = n_boot
        self.eps = eps
        self.seed = seed
        self._cache: dict[int, np.ndarray] = {}

    def null(self, n_current: int) -> np.ndarray:
        """Bootstrap PSI values under no drift for current windows of ``n_current`` rows."""
        n_current = int(n_current)
        if n_current not in self._cache:
            if self.kind == "numeric":
                self._cache[n_current] = psi_null_distribution(
                    self.reference, n_current, self.n_bins, self.n_boot, self.eps, self.seed
                )
            else:
                self._cache[n_current] = psi_null_categorical(
                    self.reference, n_current, self.n_boot, self.eps, self.seed
                )
        return self._cache[n_current]

    def threshold(self, n_current: int, alpha: float = 0.05) -> float:
        """PSI value exceeded with probability ``alpha`` when there is no drift."""
        return float(np.quantile(self.null(n_current), 1.0 - alpha))

    def p_value(self, observed: float, n_current: int) -> float:
        """Bootstrap p-value of an observed PSI for a window of ``n_current`` rows."""
        return bootstrap_p_value(self.null(n_current), observed)
