"""Shared synthetic data for the tests (no network, no downloaded data)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")


def make_frame(n: int, seed: int, shift: float = 0.0, missing: float = 0.0) -> pd.DataFrame:
    """Two numeric features, one categorical, a label and a 'prediction'."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(shift, 1.0, n)
    x2 = rng.lognormal(0.0, 0.5, n)
    cat = rng.choice(np.array(["a", "b", "c"], dtype=object), n, p=[0.5, 0.3, 0.2])
    y = 2.0 * x1 + x2 + rng.normal(0, 0.5, n)
    pred = 2.0 * x1 + x2
    x2 = np.where(rng.random(n) < missing, np.nan, x2)
    return pd.DataFrame({"x1": x1, "x2": x2, "segment": cat, "y": y, "pred": pred})


@pytest.fixture()
def reference() -> pd.DataFrame:
    return make_frame(1500, seed=1)
