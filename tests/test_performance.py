import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from driftwatch.performance import mae, performance_check, rmse, rolling_metrics


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=200), rng.normal(size=200)
    assert mae(y, p) == pytest.approx(mean_absolute_error(y, p))
    assert rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)))


def test_performance_check_flags_real_degradation_only():
    rng = np.random.default_rng(1)
    y_ref = rng.normal(size=2000)
    p_ref = y_ref + rng.normal(0, 1.0, 2000)
    y_cur = rng.normal(size=800)
    same = performance_check(y_ref, p_ref, y_cur, y_cur + rng.normal(0, 1.0, 800))
    worse = performance_check(y_ref, p_ref, y_cur, y_cur + rng.normal(0, 1.5, 800))
    assert not same.degraded and 0.9 < same.mae_ratio < 1.1
    assert worse.degraded and worse.mae_ratio == pytest.approx(1.5, abs=0.1)
    slightly = performance_check(y_ref, p_ref, y_cur, y_cur + rng.normal(0, 1.05, 800))
    assert not slightly.degraded  # +5% is inside the 10% tolerance


def test_rolling_metrics_window():
    y = np.zeros(10)
    p = np.r_[np.ones(5), 3 * np.ones(5)]
    roll = rolling_metrics(y, p, window=5)
    assert len(roll) == 6
    assert roll["mae"].iloc[0] == 1.0 and roll["mae"].iloc[-1] == 3.0
