"""Turn a set of alarms into a likely cause and next steps.

Pipeline faults (a stuck column, a unit change, a missing-value spike, a flood of unseen
categories) look like drift but need an engineering fix, not retraining; real distribution
shift needs a performance check before anyone retrains. These rules make that first call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from driftwatch.univariate import finite_values, non_missing

SCALE_FACTOR = 5.0  # median magnitude changed by >= 5x (or <= 1/5): looks like a unit change
UNSEEN_SHARE = 0.10  # >= 10% of current values are categories never seen in the reference
MIN_ROWS = 10


@dataclass
class Triage:
    """Likely cause of the alarms and suggested next steps."""

    kind: str  # ok | pipeline_fault | shift_with_damage | shift_no_damage | shift_unknown_impact
    #            | concept_shift
    headline: str
    actions: list[str] = field(default_factory=list)


def column_issues(reference: pd.Series, current: pd.Series, kind: str) -> list[str]:
    """Signatures of a broken data pipeline in one column (empty list if none)."""
    issues = []
    if kind == "numeric":
        r, c = finite_values(reference), finite_values(current)
        if c.size >= MIN_ROWS and np.unique(r).size > 1 and np.unique(c).size == 1:
            issues.append(f"stuck at a constant value ({c[0]:g})")
        elif r.size and c.size:
            mr, mc = float(np.median(np.abs(r))), float(np.median(np.abs(c)))
            if mr > 0 and mc > 0 and not (1 / SCALE_FACTOR < mc / mr < SCALE_FACTOR):
                issues.append(f"typical magnitude changed {mc / mr:.3g}x (unit change?)")
    else:
        r = set(pd.Series(non_missing(reference)).astype(str))
        c = pd.Series(non_missing(current)).astype(str)
        if len(c) >= MIN_ROWS:
            unseen = float((~c.isin(r)).mean())
            if unseen >= UNSEEN_SHARE:
                issues.append(f"{unseen:.0%} of values are categories unseen in the reference")
    return issues


def triage(report: object) -> Triage:
    """Classify a :class:`~driftwatch.monitor.DriftReport` into one actionable situation."""
    issues = report.pipeline_issues
    perf = report.performance
    if issues:
        cols = sorted({i.split(":")[0] for i in issues})
        return Triage(
            "pipeline_fault",
            f"Likely a data-pipeline fault in {', '.join(cols)}",
            [
                "Check the upstream source and transformations of the listed columns "
                "(schema, units, joins, default values).",
                "Do not retrain on this data: the model would learn the bug.",
                "Re-run this check after the fix; remaining drift is then genuine.",
            ],
        )
    if report.drift_detected:
        if perf is None:
            return Triage(
                "shift_unknown_impact",
                "Distribution shift; its effect on accuracy is unknown until labels arrive",
                [
                    "Inspect the top drifted columns and the domain-classifier importances.",
                    "Label a sample of current rows (or wait for labels) and re-check "
                    "performance before retraining.",
                ],
            )
        if perf.degraded:
            return Triage(
                "shift_with_damage",
                f"Distribution shift that hurts the model (MAE {perf.mae_ratio:.2f}x reference)",
                [
                    "Retrain or recalibrate on recent labelled data that covers the new "
                    "population, and validate on it before release.",
                    "Until then, consider a fallback for the most shifted segments.",
                ],
            )
        return Triage(
            "shift_no_damage",
            f"Distribution shift without significant damage (MAE {perf.mae_ratio:.2f}x "
            "reference)",
            [
                "No retraining needed now; keep monitoring performance as labels arrive.",
                "If alarms of this size keep proving harmless, raise min_psi.",
            ],
        )
    if report.performance_degraded or report.label_drift:
        return Triage(
            "concept_shift",
            "Inputs look stable but labels or errors moved: concept shift",
            [
                "Input-drift monitoring cannot see this; retrain on recent labels.",
                "Check whether the label definition or the labelling process changed.",
            ],
        )
    return Triage("ok", "No action needed", [])
