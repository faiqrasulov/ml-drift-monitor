import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_frame

from driftwatch import DriftConfig, DriftMonitor, compare


def _monitor(reference, **cfg):
    config = DriftConfig(n_bootstrap=300, **cfg)
    return DriftMonitor(config).fit(reference, target="y", prediction="pred")


def test_no_drift_raises_no_alarm(reference):
    report = _monitor(reference).check(make_frame(600, seed=2))
    assert not report.drift_detected
    assert not report.alarm
    assert {c.column for c in report.columns} == {"x1", "x2", "segment", "pred"}
    assert report.label is not None and report.label.column == "y"
    assert 0.4 < report.domain.auc < 0.6


def test_shift_is_detected_and_localised(reference):
    report = _monitor(reference).check(make_frame(600, seed=3, shift=0.5))
    drifted = set(report.drifted_columns)
    assert {"x1", "pred"} <= drifted
    assert "x2" not in drifted and "segment" not in drifted
    assert report.domain_drift and report.alarm
    assert report.domain.importances["x1"] == max(report.domain.importances.values())


def test_missing_spike_flags_missing_rate(reference):
    report = _monitor(reference).check(make_frame(600, seed=4, missing=0.3))
    x2 = next(c for c in report.columns if c.column == "x2")
    assert x2.missing_drift and x2.missing_current == pytest.approx(0.3, abs=0.06)
    assert not x2.drift  # the observed values themselves did not change


def test_concept_shift_is_invisible_to_inputs_but_caught_by_performance(reference):
    cur = make_frame(600, seed=5)
    cur["y"] = cur["y"] + 1.5  # same inputs and predictions, different labels
    report = _monitor(reference).check(cur)
    assert not report.drift_detected
    assert report.label_drift
    assert report.performance_degraded and report.performance.mae_ratio > 1.5


def test_fixed_threshold_and_ks_modes(reference):
    cur = make_frame(600, seed=6, shift=0.5)
    rule = _monitor(reference, psi_threshold=0.2, domain_classifier=False).check(cur)
    x1 = next(c for c in rule.columns if c.column == "x1")
    assert x1.test == "psi > 0.2" and x1.p_value is None and x1.drift == (x1.psi > 0.2)
    ks = _monitor(reference, test="ks", domain_classifier=False).check(cur)
    tests = {c.column: c.test for c in ks.columns}
    assert tests["x1"] == "ks" and tests["segment"] == "chi2"
    assert "x1" in ks.drifted_columns


def test_min_psi_floor_suppresses_tiny_but_significant_drift(reference):
    big = make_frame(20000, seed=7, shift=0.08)  # large window: tiny shift is significant
    loose = _monitor(reference, domain_classifier=False).check(big)
    strict = _monitor(reference, domain_classifier=False, min_psi=0.1).check(big)
    assert "x1" in loose.drifted_columns
    assert "x1" not in strict.drifted_columns


def test_report_serialises_to_json_and_frame(reference, tmp_path):
    report = compare(reference, make_frame(400, seed=8, shift=1.0), target="y",
                     prediction="pred", config=DriftConfig(n_bootstrap=200))
    report.to_json(tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["alarm"] is True and "x1" in data["drifted_columns"]
    assert data["domain_classifier"]["auc"] > 0.7
    frame = report.to_frame()
    assert len(frame) == 5 and "p_adjusted" in frame.columns


def test_string_dtype_and_validation(reference):
    ref = reference.copy()
    ref["segment"] = ref["segment"].astype("string")
    cur = make_frame(300, seed=9)
    cur["segment"] = cur["segment"].astype("string")
    report = _monitor(ref).check(cur)
    seg = next(c for c in report.columns if c.column == "segment")
    assert seg.kind == "categorical" and seg.js is not None
    with pytest.raises(KeyError):
        _monitor(ref).check(cur.drop(columns=["x1"]))
    with pytest.raises(ValueError):
        DriftConfig(correction="holm")
    with pytest.raises(ValueError):
        DriftConfig(test="ks", psi_threshold=0.2)
    with pytest.raises(RuntimeError):
        DriftMonitor().check(cur)
    assert isinstance(report.to_frame(), pd.DataFrame)
    assert np.isfinite(seg.psi)
