"""DriftMonitor: fit on reference data once, then check any number of current windows."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from driftwatch import __version__
from driftwatch.calibration import PSICalibrator
from driftwatch.multiple_testing import CORRECTIONS, adjust
from driftwatch.multivariate import DomainClassifierResult, domain_classifier_test
from driftwatch.performance import PerformanceResult, performance_check
from driftwatch.triage import Triage, column_issues, triage
from driftwatch.univariate import (
    chi2_homogeneity,
    finite_values,
    js_distance_categorical,
    ks_test,
    missing_rate_test,
    non_missing,
    normalized_wasserstein,
    psi,
    psi_categorical,
    quantile_bin_edges,
)


@dataclass
class DriftConfig:
    """How drift is decided.

    test: "psi" uses PSI with a bootstrap-calibrated p-value per column (default), "ks" uses
        the two-sample KS test for numeric and chi-square for categorical columns.
    psi_threshold: if set (e.g. 0.2, the common rule of thumb), a column drifts when its PSI
        exceeds this fixed value instead of when its calibrated p-value is small.
    correction: multiple-testing correction across all drift tests of one check
        ("bh" = Benjamini-Hochberg, "bonferroni", or "none").
    min_psi: practical-significance floor; a statistically significant column only counts as
        drifted if its PSI is at least this large (0 = off).
    max_mae_ratio: performance alarm when MAE is significantly above this multiple of the
        reference MAE (needs labels and predictions in both samples).
    """

    alpha: float = 0.05
    correction: str = "bh"
    test: str = "psi"
    psi_threshold: float | None = None
    n_bins: int = 10
    eps: float = 1e-4
    n_bootstrap: int = 1000
    min_psi: float = 0.0
    domain_classifier: bool = True
    domain_splits: int = 2
    domain_max_samples: int = 5000
    domain_importance: bool = True
    max_mae_ratio: float = 1.10
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.correction not in CORRECTIONS:
            raise ValueError(f"correction must be one of {CORRECTIONS}")
        if self.test not in ("psi", "ks"):
            raise ValueError("test must be 'psi' or 'ks'")
        if self.psi_threshold is not None and self.test != "psi":
            raise ValueError("psi_threshold only applies to test='psi'")
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")


@dataclass
class ColumnDrift:
    """All drift statistics for one column and the resulting decision."""

    column: str
    role: str  # "feature", "prediction" or "target"
    kind: str  # "numeric" or "categorical"
    n_reference: int
    n_current: int
    psi: float
    ks: float | None
    ks_p: float | None
    wasserstein: float | None  # normalised by the reference SD
    js: float | None
    chi2_p: float | None
    missing_reference: float
    missing_current: float
    missing_p: float
    test: str
    p_value: float | None
    p_adjusted: float | None = None
    missing_p_adjusted: float | None = None
    drift: bool = False
    missing_drift: bool = False
    issues: list[str] = field(default_factory=list)  # pipeline-fault signatures


@dataclass
class DriftReport:
    """Result of one check of a current window against the reference."""

    columns: list[ColumnDrift]
    domain: DomainClassifierResult | None
    domain_p_adjusted: float | None
    domain_drift: bool
    label: ColumnDrift | None
    performance: PerformanceResult | None
    config: DriftConfig
    n_reference: int
    n_current: int
    reference_name: str = "reference"
    current_name: str = "current"
    reference_data: pd.DataFrame | None = field(default=None, repr=False)
    current_data: pd.DataFrame | None = field(default=None, repr=False)
    time_column: str | None = None

    @property
    def drifted_columns(self) -> list[str]:
        return [c.column for c in self.columns if c.drift or c.missing_drift]

    @property
    def drift_detected(self) -> bool:
        """Input or prediction drift: any column, missing-rate or domain-classifier alarm."""
        return bool(self.drifted_columns) or self.domain_drift

    @property
    def label_drift(self) -> bool:
        return bool(self.label is not None and self.label.drift)

    @property
    def performance_degraded(self) -> bool:
        return bool(self.performance is not None and self.performance.degraded)

    @property
    def pipeline_issues(self) -> list[str]:
        """Pipeline-fault signatures found in any column, as "column: issue" strings."""
        cols = list(self.columns) + ([self.label] if self.label is not None else [])
        return [f"{c.column}: {issue}" for c in cols for issue in c.issues]

    @property
    def triage(self) -> Triage:
        """Likely cause of the alarms and suggested next steps."""
        return triage(self)

    @property
    def alarm(self) -> bool:
        """True if anything fired; the CLI exits with code 1 in that case."""
        return self.drift_detected or self.label_drift or self.performance_degraded

    def alarms(self) -> list[str]:
        """Human-readable list of everything that fired."""
        out = []
        for c in self.columns:
            if c.drift:
                out.append(f"{c.role} '{c.column}' drifted ({c.test}, PSI = {c.psi:.3f})")
            if c.missing_drift:
                out.append(
                    f"missing rate of '{c.column}' changed "
                    f"({c.missing_reference:.1%} -> {c.missing_current:.1%})"
                )
        if self.domain_drift and self.domain is not None:
            out.append(f"domain classifier separates the samples (AUC = {self.domain.auc:.3f})")
        if self.label_drift and self.label is not None:
            out.append(f"label '{self.label.column}' drifted (PSI = {self.label.psi:.3f})")
        if self.performance_degraded and self.performance is not None:
            out.append(
                f"MAE degraded to {self.performance.mae_ratio:.2f}x the reference "
                f"(tolerance {self.performance.max_ratio:.2f}x)"
            )
        return out

    def to_frame(self) -> pd.DataFrame:
        """Per-column results as a DataFrame (one row per monitored column)."""
        rows = [asdict(c) for c in self.columns]
        if self.label is not None:
            rows.append(asdict(self.label))
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        """JSON-serialisable summary (NaN and inf become null)."""
        perf = None
        if self.performance is not None:
            perf = asdict(self.performance)
            perf["mae_ratio"] = self.performance.mae_ratio
            perf["rmse_ratio"] = self.performance.rmse_ratio
        dom = None
        if self.domain is not None:
            dom = asdict(self.domain)
            dom["p_adjusted"] = self.domain_p_adjusted
            dom["drift"] = self.domain_drift
        out = {
            "driftwatch_version": __version__,
            "reference": {"name": self.reference_name, "rows": self.n_reference},
            "current": {"name": self.current_name, "rows": self.n_current},
            "alarm": self.alarm,
            "drift_detected": self.drift_detected,
            "label_drift": self.label_drift,
            "performance_degraded": self.performance_degraded,
            "alarms": self.alarms(),
            "triage": asdict(self.triage),
            "pipeline_issues": self.pipeline_issues,
            "drifted_columns": self.drifted_columns,
            "columns": [asdict(c) for c in self.columns],
            "label": asdict(self.label) if self.label is not None else None,
            "domain_classifier": dom,
            "performance": perf,
            "config": asdict(self.config),
        }
        return _jsonable(out)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def to_html(self, path: str | Path, title: str | None = None) -> None:
        """Write a self-contained HTML report (inline CSS, embedded PNG charts)."""
        from driftwatch.report import render_html

        Path(path).write_text(render_html(self, title=title), encoding="utf-8")


def _jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def infer_categorical(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Columns that are not numeric (strings, categoricals, booleans) are treated as categorical."""
    out = []
    for c in columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            out.append(c)
    return out


class DriftMonitor:
    """Data, prediction and performance drift monitor.

    ``fit`` stores the reference sample and prepares calibrated thresholds; ``check`` compares
    a current window against it. Bootstrap null distributions are cached per window size, so
    repeated checks of same-sized windows are cheap.
    """

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()
        self.reference: pd.DataFrame | None = None

    def fit(
        self,
        reference: pd.DataFrame,
        features: Sequence[str] | None = None,
        categorical: Sequence[str] | None = None,
        target: str | None = None,
        prediction: str | None = None,
        time_column: str | None = None,
        name: str = "reference",
    ) -> DriftMonitor:
        """Store the reference sample and the roles of its columns.

        ``time_column`` (optional) is excluded from the drift tests and used to order the current
        rows for the rolling-performance chart.
        """
        for col in [target, prediction]:
            if col is not None and col not in reference.columns:
                raise KeyError(f"column {col!r} not in reference data")
        special = {target, prediction, time_column} - {None}
        self.features = list(features or [c for c in reference.columns if c not in special])
        missing = [c for c in self.features if c not in reference.columns]
        if missing:
            raise KeyError(f"features not in reference data: {missing}")
        monitored = self.features + [c for c in (prediction, target) if c is not None]
        self.categorical = list(
            categorical if categorical is not None else infer_categorical(reference, monitored)
        )
        self.target = target
        self.prediction = prediction
        self.time_column = time_column
        self.reference = reference.reset_index(drop=True)
        self.name = name
        self._calibrators: dict[str, PSICalibrator] = {}
        self._edges: dict[str, np.ndarray] = {}
        for i, col in enumerate(monitored):
            kind = "categorical" if col in self.categorical else "numeric"
            if kind == "numeric":
                self._edges[col] = quantile_bin_edges(self.reference[col], self.config.n_bins)
            self._calibrators[col] = PSICalibrator(
                self.reference[col],
                kind=kind,
                n_bins=self.config.n_bins,
                n_boot=self.config.n_bootstrap,
                eps=self.config.eps,
                seed=self.config.seed + i,
            )
        return self

    # ----------------------------------------------------------------------- per column

    def _column(self, col: str, role: str, current: pd.DataFrame) -> ColumnDrift:
        cfg = self.config
        ref, cur = self.reference[col], current[col]
        kind = "categorical" if col in self.categorical else "numeric"
        miss = missing_rate_test(ref, cur)
        n_cur = int(len(non_missing(cur)))
        if kind == "numeric":
            ref_v, cur_v = finite_values(ref), finite_values(cur)
            value = psi(ref_v, cur_v, eps=cfg.eps, edges=self._edges[col]) if n_cur else np.nan
            ks = ks_test(ref_v, cur_v) if n_cur else None
            wd = normalized_wasserstein(ref_v, cur_v) if n_cur else None
            js = chi2_p = None
        else:
            value = psi_categorical(ref, cur, eps=cfg.eps) if n_cur else np.nan
            ks = wd = None
            js = js_distance_categorical(ref, cur) if n_cur else None
            chi2_p = chi2_homogeneity(ref, cur).p_value if n_cur else None
        if n_cur == 0:
            test, p = "none (no values)", None
        elif cfg.test == "ks":
            test = "ks" if kind == "numeric" else "chi2"
            p = ks.p_value if kind == "numeric" else chi2_p
        elif cfg.psi_threshold is not None:
            test, p = f"psi > {cfg.psi_threshold:g}", None
        else:
            test, p = "psi (bootstrap)", self._calibrators[col].p_value(value, n_cur)
        return ColumnDrift(
            column=col,
            role=role,
            kind=kind,
            n_reference=int(len(non_missing(ref))),
            n_current=n_cur,
            psi=float(value),
            ks=None if ks is None else ks.statistic,
            ks_p=None if ks is None else ks.p_value,
            wasserstein=wd,
            js=js,
            chi2_p=chi2_p,
            missing_reference=miss.reference_rate,
            missing_current=miss.current_rate,
            missing_p=miss.p_value,
            test=test,
            p_value=p,
            issues=column_issues(ref, cur, kind),
        )

    # ----------------------------------------------------------------------- check

    def check(
        self, current: pd.DataFrame, name: str = "current", domain_classifier: bool | None = None
    ) -> DriftReport:
        """Compare ``current`` to the reference; returns a :class:`DriftReport`."""
        if self.reference is None:
            raise RuntimeError("call fit() before check()")
        cfg = self.config
        missing = [c for c in self.features if c not in current.columns]
        if missing:
            raise KeyError(f"features not in current data: {missing}")
        if len(current) == 0:
            raise ValueError("current data is empty")
        if self.time_column is not None and self.time_column in current.columns:
            current = current.sort_values(self.time_column, kind="stable")
        current = current.reset_index(drop=True)
        roles = [(c, "feature") for c in self.features]
        if self.prediction is not None and self.prediction in current.columns:
            roles.append((self.prediction, "prediction"))
        cols = [self._column(c, r, current) for c, r in roles]

        use_domain = cfg.domain_classifier if domain_classifier is None else domain_classifier
        domain = None
        if use_domain and len(self.features) > 0:
            domain = domain_classifier_test(
                self.reference,
                current,
                self.features,
                [c for c in self.categorical if c in self.features],
                n_splits=cfg.domain_splits,
                max_samples=cfg.domain_max_samples,
                importance=cfg.domain_importance,
                seed=cfg.seed,
            )

        # One family of hypotheses per check: every column's primary test, every missing-rate
        # test where missing values occur, and the domain classifier.
        family: list[tuple[str, int]] = []
        pvals: list[float] = []
        for i, c in enumerate(cols):
            if c.p_value is not None:
                family.append(("primary", i))
                pvals.append(c.p_value)
            if c.missing_reference > 0 or c.missing_current > 0:
                family.append(("missing", i))
                pvals.append(c.missing_p)
        if domain is not None:
            family.append(("domain", -1))
            pvals.append(domain.p_value)
        adjusted = adjust(pvals, cfg.correction) if pvals else np.array([])
        domain_adj = None
        for (what, i), adj in zip(family, adjusted, strict=True):
            if what == "primary":
                cols[i].p_adjusted = float(adj)
            elif what == "missing":
                c = cols[i]
                c.missing_p_adjusted = float(adj)
                c.missing_drift = bool(adj <= cfg.alpha)
                if c.missing_drift and c.missing_current > c.missing_reference:
                    c.issues.append(
                        f"missing rate jumped {c.missing_reference:.1%} -> {c.missing_current:.1%}"
                    )
            else:
                domain_adj = float(adj)
        for c in cols:
            if cfg.psi_threshold is not None and cfg.test == "psi":
                c.drift = bool(c.psi > cfg.psi_threshold)
            elif c.p_adjusted is not None:
                c.drift = bool(c.p_adjusted <= cfg.alpha and c.psi >= cfg.min_psi)

        label = None
        if self.target is not None and self.target in current.columns:
            label = self._column(self.target, "target", current)
            if cfg.psi_threshold is not None and cfg.test == "psi":
                label.drift = bool(label.psi > cfg.psi_threshold)
            elif label.p_value is not None:
                label.p_adjusted = label.p_value
                label.drift = bool(label.p_value <= cfg.alpha and label.psi >= cfg.min_psi)

        perf = None
        if (
            self.target is not None
            and self.prediction is not None
            and {self.target, self.prediction} <= set(current.columns)
        ):
            perf = performance_check(
                self.reference[self.target],
                self.reference[self.prediction],
                current[self.target],
                current[self.prediction],
                max_ratio=cfg.max_mae_ratio,
                alpha=cfg.alpha,
            )

        return DriftReport(
            columns=cols,
            domain=domain,
            domain_p_adjusted=domain_adj,
            domain_drift=bool(domain_adj is not None and domain_adj <= cfg.alpha),
            label=label,
            performance=perf,
            config=cfg,
            n_reference=len(self.reference),
            n_current=len(current),
            reference_name=self.name,
            current_name=name,
            reference_data=self.reference,
            current_data=current,
            time_column=self.time_column if self.time_column in current.columns else None,
        )


def compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target: str | None = None,
    prediction: str | None = None,
    features: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    time_column: str | None = None,
    config: DriftConfig | None = None,
) -> DriftReport:
    """One-shot drift check of ``current`` against ``reference``."""
    monitor = DriftMonitor(config).fit(
        reference, features=features, categorical=categorical, target=target,
        prediction=prediction, time_column=time_column,
    )
    return monitor.check(current)
