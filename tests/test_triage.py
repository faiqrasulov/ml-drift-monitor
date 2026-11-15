import numpy as np
import pandas as pd
from conftest import make_frame

from driftwatch import DriftConfig, DriftMonitor
from driftwatch.triage import column_issues


def test_column_issue_signatures():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.lognormal(1.0, 0.4, 500))
    cur = pd.Series(rng.lognormal(1.0, 0.4, 200))
    assert column_issues(ref, cur, "numeric") == []
    assert "stuck" in column_issues(ref, pd.Series(np.full(50, 3.0)), "numeric")[0]
    assert "unit change" in column_issues(ref, cur * 1000, "numeric")[0]
    assert "unit change" in column_issues(ref, cur / 1000, "numeric")[0]
    assert column_issues(ref, cur * 1.5, "numeric") == []  # a real but moderate shift
    cats = pd.Series(["a", "b"] * 50)
    assert column_issues(cats, pd.Series(["a"] * 80 + ["z"] * 20), "categorical")
    assert column_issues(cats, pd.Series(["a"] * 95 + ["z"] * 5), "categorical") == []


def _check(reference, current):
    cfg = DriftConfig(n_bootstrap=300)
    return DriftMonitor(cfg).fit(reference, target="y", prediction="pred").check(current)


def test_triage_kinds(reference):
    assert _check(reference, make_frame(600, seed=2)).triage.kind == "ok"
    harmless = _check(reference, make_frame(600, seed=3, shift=0.5))
    assert harmless.triage.kind == "shift_no_damage"  # model is right on the new inputs too
    missing = _check(reference, make_frame(600, seed=4, missing=0.3))
    assert missing.triage.kind == "pipeline_fault"
    assert any("missing rate jumped" in i for i in missing.pipeline_issues)
    stuck = make_frame(600, seed=5)
    stuck["x2"] = 1.0
    assert _check(reference, stuck).triage.kind == "pipeline_fault"
    concept = make_frame(600, seed=6)
    concept["y"] = concept["y"] + 1.5
    assert _check(reference, concept).triage.kind == "concept_shift"
    damaging = make_frame(600, seed=7, shift=0.5)
    damaging["y"] = damaging["y"] + 2.0 * damaging["x1"]  # shift + changed relationship
    rep = _check(reference, damaging)
    assert rep.triage.kind == "shift_with_damage" and rep.triage.actions
    no_labels = make_frame(600, seed=8, shift=0.5).drop(columns=["y"])
    assert _check(reference, no_labels).triage.kind == "shift_unknown_impact"
