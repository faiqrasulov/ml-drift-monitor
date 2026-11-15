import numpy as np
import pandas as pd

from driftwatch.multivariate import auc_null_p_value, domain_classifier_test, encode_frame


def _frame(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(shift, 1.0, n),
        "c": rng.choice(np.array(["x", "y", "z"], dtype=object), n),
    })


def test_auc_about_half_on_same_distribution():
    res = domain_classifier_test(_frame(1200, 1), _frame(600, 2), ["a", "b", "c"], ["c"])
    assert abs(res.auc - 0.5) < 0.05
    assert res.p_value > 0.01
    assert len(res.fold_auc) == 2


def test_auc_high_on_shifted_data_and_importance_points_at_shifted_column():
    res = domain_classifier_test(_frame(1200, 1), _frame(600, 3, shift=2.0), ["a", "b", "c"], ["c"])
    assert res.auc > 0.85
    assert res.p_value < 1e-6
    assert max(res.importances, key=res.importances.get) == "b"
    assert res.importances["b"] > 5 * max(abs(res.importances["a"]), abs(res.importances["c"]))


def test_encode_frame_keeps_missing_and_codes_categories():
    ref = pd.DataFrame({"n": [1.0, np.nan], "c": ["u", None]})
    cur = pd.DataFrame({"n": [3.0], "c": ["v"]})
    X, y, is_cat = encode_frame(ref, cur, ["n", "c"], ["c"])
    assert X.shape == (3, 2) and list(y) == [0, 0, 1] and list(is_cat) == [False, True]
    assert np.isnan(X[1, 0]) and np.isnan(X[1, 1])
    assert X[0, 1] != X[2, 1]


def test_auc_null_p_value_symmetry():
    assert auc_null_p_value(0.5, 100, 100) == 0.5
    assert auc_null_p_value(0.6, 1000, 1000) < 1e-10
