"""Model performance monitoring once labels arrive (regression: MAE and RMSE)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from driftwatch.univariate import normal_sf


@dataclass
class PerformanceResult:
    """Current vs reference error, and a one-sided test that MAE grew by more than a margin.

    ``p_value`` tests H0: MAE_current <= max_ratio * MAE_reference against the alternative that
    the model is worse than that, using a Welch-type z statistic on absolute errors. A small
    p-value therefore means "degraded by more than the tolerated margin, beyond sampling noise".
    """

    n_reference: int
    n_current: int
    mae_reference: float
    mae_current: float
    rmse_reference: float
    rmse_current: float
    max_ratio: float
    p_value: float
    degraded: bool

    @property
    def mae_ratio(self) -> float:
        return self.mae_current / self.mae_reference

    @property
    def rmse_ratio(self) -> float:
        return self.rmse_current / self.rmse_reference


def _clean_pairs(y: object, pred: object) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if y.shape != pred.shape:
        raise ValueError("y and pred must have the same length")
    ok = ~(np.isnan(y) | np.isnan(pred))
    return y[ok], pred[ok]


def mae(y: object, pred: object) -> float:
    """Mean absolute error, ignoring rows where the label or prediction is missing."""
    y, pred = _clean_pairs(y, pred)
    return float(np.mean(np.abs(y - pred)))


def rmse(y: object, pred: object) -> float:
    """Root mean squared error, ignoring rows where the label or prediction is missing."""
    y, pred = _clean_pairs(y, pred)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def performance_check(
    y_reference: object,
    pred_reference: object,
    y_current: object,
    pred_current: object,
    max_ratio: float = 1.10,
    alpha: float = 0.05,
) -> PerformanceResult:
    """Compare current to reference error; flag degradation beyond ``max_ratio`` x MAE.

    The reference errors must be out-of-sample (a validation set, not the training data),
    otherwise the reference MAE is optimistic and every window looks degraded.
    """
    yr, pr = _clean_pairs(y_reference, pred_reference)
    yc, pc = _clean_pairs(y_current, pred_current)
    if yr.size < 2 or yc.size < 2:
        raise ValueError("need at least 2 labelled rows in each sample")
    er, ec = np.abs(yr - pr), np.abs(yc - pc)
    diff = ec.mean() - max_ratio * er.mean()
    se = math.sqrt(ec.var(ddof=1) / ec.size + max_ratio**2 * er.var(ddof=1) / er.size)
    p = normal_sf(diff / se) if se > 0 else float(diff <= 0)
    return PerformanceResult(
        n_reference=int(yr.size),
        n_current=int(yc.size),
        mae_reference=float(er.mean()),
        mae_current=float(ec.mean()),
        rmse_reference=float(np.sqrt(np.mean(er**2))),
        rmse_current=float(np.sqrt(np.mean(ec**2))),
        max_ratio=max_ratio,
        p_value=float(p),
        degraded=bool(p < alpha),
    )


def rolling_metrics(y: object, pred: object, window: int) -> pd.DataFrame:
    """Rolling MAE and RMSE over consecutive rows (rows assumed to be in time order)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    err = pd.Series(y - pred)
    out = pd.DataFrame(
        {
            "mae": err.abs().rolling(window, min_periods=window).mean(),
            "rmse": np.sqrt((err**2).rolling(window, min_periods=window).mean()),
        }
    )
    out.index.name = "row"
    return out.dropna()
