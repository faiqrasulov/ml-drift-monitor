"""Multiple-testing corrections across many per-feature drift tests."""

from __future__ import annotations

import numpy as np

CORRECTIONS = ("bh", "bonferroni", "none")


def bonferroni(p_values: object) -> np.ndarray:
    """Bonferroni-adjusted p-values min(m * p, 1): controls the family-wise error rate."""
    p = np.asarray(p_values, dtype=np.float64)
    return np.minimum(p * p.size, 1.0)


def benjamini_hochberg(p_values: object) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values: controls the false discovery rate.

    adj_(i) = min_{j >= i} m * p_(j) / j over the sorted p-values, capped at 1.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted_sorted, 1.0)
    return out


def adjust(p_values: object, method: str = "bh") -> np.ndarray:
    """Adjust p-values with ``method`` in {"bh", "bonferroni", "none"}."""
    if method == "bh":
        return benjamini_hochberg(p_values)
    if method == "bonferroni":
        return bonferroni(p_values)
    if method == "none":
        return np.asarray(p_values, dtype=np.float64).copy()
    raise ValueError(f"unknown correction {method!r}; choose from {CORRECTIONS}")


def reject(p_values: object, alpha: float = 0.05, method: str = "bh") -> np.ndarray:
    """Boolean mask of hypotheses rejected at level ``alpha`` after the correction."""
    return adjust(p_values, method) <= alpha
