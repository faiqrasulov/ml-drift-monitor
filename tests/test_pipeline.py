"""End-to-end smoke test of the experiment pipeline on tiny synthetic California-like data."""

import json

import numpy as np
import pandas as pd

from driftwatch.data import FEATURES, TARGET, split_regions
from driftwatch.experiments import run
from driftwatch.scenarios import DETECTORS, SCENARIOS


def synthetic_california(n: int = 2400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    south = rng.random(n) < 0.6
    lat = np.where(south, rng.uniform(32.5, 35.0, n), rng.uniform(35.0, 42.0, n))
    med_inc = rng.lognormal(1.2 - 0.04 * (lat - 35), 0.45, n)
    df = pd.DataFrame({
        "MedInc": med_inc,
        "HouseAge": rng.integers(1, 53, n).astype(float),
        "AveRooms": rng.lognormal(1.6, 0.25, n),
        "AveBedrms": rng.lognormal(0.05, 0.1, n),
        "Population": rng.lognormal(7.0, 0.7, n),
        "AveOccup": rng.lognormal(1.05, 0.3, n),
        "Latitude": lat,
        "Longitude": rng.uniform(-124.0, -114.0, n),
    })
    df[TARGET] = np.clip(0.45 * med_inc + 0.004 * df["HouseAge"] + rng.normal(0, 0.4, n), 0.15, 5)
    return df


def test_split_regions_is_disjoint_and_complete():
    df = synthetic_california()
    s = split_regions(df)
    assert len(s.train) + len(s.reference) + len(s.holdout) + len(s.north) == len(df)
    assert (s.north["Latitude"] >= 35).all() and (s.holdout["Latitude"] < 35).all()


def test_pipeline_end_to_end(tmp_path):
    out = tmp_path / "reports"
    m = run(out, tmp_path / "data", scale=0.02, window=100, df=synthetic_california())
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["north_vs_south"]["mae_ratio"] == m["north_vs_south"]["mae_ratio"]
    res = saved["scenarios"]["results"]
    assert set(res) == {s.key for s in SCENARIOS}
    for r in res.values():
        assert set(r["alarm_rate"]) == {k for k, _ in DETECTORS}
        assert all(0.0 <= v <= 1.0 for v in r["alarm_rate"].values())
    # a 10x unit change on the most important feature must be caught and must hurt
    assert res["medinc_units"]["alarm_rate"]["driftwatch_alarm"] == 1.0
    assert res["medinc_units"]["mae_ratio_mean"] > 1.5
    # a pure label shift leaves the inputs alone
    assert res["concept_shift"]["alarm_rate"]["missing_rate"] == 0.0
    assert set(saved["psi_false_alarm"]["any_feature"]) >= {"ks_bh", "psi_gt_0.2_bins10"}
    figures = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert figures == ["detector_heatmap.png", "drift_vs_performance.png",
                       "north_vs_south_dashboard.png", "psi_false_alarms.png"]
    for name in ("reference", "north", "holdout"):
        assert (tmp_path / "data" / "demo" / f"{name}.parquet").exists()
    assert list(FEATURES)[0] == "MedInc"
