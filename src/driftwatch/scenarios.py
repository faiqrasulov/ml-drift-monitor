"""Scenario definitions and synthetic drift injections for the California Housing study.

Injections are applied to real held-out rows so the ground truth is known. Each function
returns a modified copy; the input frame is never changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    group: str  # "none", "real", "covariate", "pipeline", "concept"
    description: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("no_drift", "No drift\n(same region)", "none",
             "Real: random rows of the held-out Southern California data"),
    Scenario("north", "North/Central CA\n(real shift)", "real",
             "Real: random rows from latitude >= 35 N"),
    Scenario("income_mild", "Income shift\nmild", "covariate",
             "Injected: held-out rows resampled towards high income (beta = 0.25)"),
    Scenario("income_strong", "Income shift\nstrong", "covariate",
             "Injected: held-out rows resampled towards high income (beta = 1.0)"),
    Scenario("houseage_stuck", "HouseAge stuck\nat constant", "pipeline",
             "Injected: HouseAge replaced by the reference median"),
    Scenario("population_units", "Population\nin thousands", "pipeline",
             "Injected: Population divided by 1,000 (unit change)"),
    Scenario("medinc_units", "MedInc x10\n(unit change)", "pipeline",
             "Injected: MedInc multiplied by 10 ($1,000s instead of $10,000s)"),
    Scenario("missing_spike", "AveOccup 30%\nmissing", "pipeline",
             "Injected: 30% of AveOccup set to missing at random"),
    Scenario("concept_shift", "Prices +20%\n(concept shift)", "concept",
             "Injected: labels x1.2, inputs unchanged"),
)

DETECTORS: tuple[tuple[str, str], ...] = (
    ("psi_gt_0.2", "PSI > 0.2, any feature (rule of thumb)"),
    ("psi_gt_0.1", "PSI > 0.1, any feature"),
    ("ks_uncorrected", "KS p < 0.05, any feature (no correction)"),
    ("ks_bh", "KS + Benjamini-Hochberg"),
    ("psi_calibrated", "Calibrated PSI + BH (driftwatch)"),
    ("missing_rate", "Missing-rate test (driftwatch)"),
    ("domain_classifier", "Domain classifier AUC test (driftwatch)"),
    ("prediction_drift", "Prediction drift, calibrated PSI (driftwatch)"),
    ("driftwatch_alarm", "driftwatch input + prediction alarm"),
    ("pipeline_triage", "Pipeline-fault signatures (driftwatch triage)"),
    ("label_drift", "Label drift (needs labels)"),
    ("performance", "MAE > 1.1x reference (needs labels)"),
)


def tilted_sample(
    pool: pd.DataFrame, n: int, beta: float, rng: np.random.Generator, column: str = "MedInc"
) -> pd.DataFrame:
    """Covariate shift: resample real rows with weights exp(beta * z), z = standardised log(column).

    Rows (and their true labels) are real; only the mix changes, towards higher values of
    ``column`` for beta > 0. beta = 0 is a plain random sample.
    """
    x = np.log(pool[column].to_numpy(dtype=np.float64))
    z = (x - x.mean()) / x.std()
    w = np.exp(beta * z)
    idx = rng.choice(len(pool), size=n, replace=False, p=w / w.sum())
    return pool.iloc[np.sort(idx)].reset_index(drop=True)


def stuck_value(df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
    """A broken pipeline that keeps sending one constant value for ``column``."""
    out = df.copy()
    out[column] = float(value)
    return out


def unit_change(df: pd.DataFrame, column: str, factor: float) -> pd.DataFrame:
    """An upstream unit change: ``column`` arrives multiplied by ``factor``."""
    out = df.copy()
    out[column] = out[column] * factor
    return out


def missing_spike(
    df: pd.DataFrame, column: str, rate: float, rng: np.random.Generator
) -> pd.DataFrame:
    """A share ``rate`` of ``column`` goes missing completely at random."""
    out = df.copy()
    mask = rng.random(len(out)) < rate
    values = out[column].to_numpy(dtype=np.float64, copy=True)
    values[mask] = np.nan
    out[column] = values
    return out


def label_shift(df: pd.DataFrame, target: str, factor: float) -> pd.DataFrame:
    """Concept shift: the same inputs now map to ``factor`` times the old label."""
    out = df.copy()
    out[target] = out[target] * factor
    return out
