"""Reproduce every number and figure in the README.

    python -m driftwatch.experiments --out reports --data data

Steps: train a price model on Southern California, then (1) compare Northern/Central
California against the reference, (2) simulate the false-alarm rate of PSI rules under no
drift, (3) run every detector on repeated 500-row windows of real and injected scenarios, and
(4) relate drift scores to the actual change in model error.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from driftwatch import __version__
from driftwatch.calibration import psi_against, psi_null_distribution, reference_bins
from driftwatch.data import FEATURES, PREDICTION, TARGET, load_california, split_regions
from driftwatch.figures import make_figures
from driftwatch.monitor import DriftConfig, DriftMonitor, DriftReport
from driftwatch.multiple_testing import benjamini_hochberg, bonferroni
from driftwatch.multivariate import auc_null_p_value, domain_classifier_test, encode_frame
from driftwatch.performance import mae, rmse
from driftwatch.scenarios import (
    DETECTORS,
    SCENARIOS,
    label_shift,
    missing_spike,
    stuck_value,
    tilted_sample,
    unit_change,
)
from driftwatch.univariate import ks_p_value, ks_statistic_sorted

SEED = 20261115
WINDOW = 500


# --------------------------------------------------------------------------- model


def train_model(train: pd.DataFrame, seed: int = 0) -> HistGradientBoostingRegressor:
    """Gradient-boosted regressor for median house value from the six census features."""
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=seed
    )
    return model.fit(train[list(FEATURES)], train[TARGET])


def permutation_importance_mae(
    model: HistGradientBoostingRegressor, df: pd.DataFrame, repeats: int = 5, seed: int = 0
) -> dict[str, float]:
    """Increase in MAE when one feature is shuffled (how much the model relies on it)."""
    rng = np.random.default_rng(seed)
    X = df[list(FEATURES)]
    base = mae(df[TARGET], model.predict(X))
    out = {}
    for f in FEATURES:
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(mae(df[TARGET], model.predict(Xp)) - base)
        out[f] = float(np.mean(drops))
    return out


def with_predictions(df: pd.DataFrame, model: HistGradientBoostingRegressor) -> pd.DataFrame:
    """Features, label and the model's prediction (computed from the features as given)."""
    out = df[list(FEATURES) + [TARGET]].copy()
    out[PREDICTION] = model.predict(out[list(FEATURES)])
    return out


# --------------------------------------------------------------------------- scenarios


def scenario_window(
    key: str,
    pools: dict[str, pd.DataFrame],
    model: HistGradientBoostingRegressor,
    rng: np.random.Generator,
    houseage_value: float,
    n: int = WINDOW,
    beta: float | None = None,
) -> pd.DataFrame:
    """One window of ``n`` rows for scenario ``key``, with the model's predictions."""
    pool = pools["north"] if key == "north" else pools["holdout"]
    if key.startswith("income") or beta is not None:
        b = beta if beta is not None else (0.25 if key == "income_mild" else 1.0)
        w = tilted_sample(pool, n, b, rng)
    else:
        w = pool.iloc[np.sort(rng.choice(len(pool), n, replace=False))].reset_index(drop=True)
    if key == "houseage_stuck":
        w = stuck_value(w, "HouseAge", houseage_value)
    elif key == "population_units":
        w = unit_change(w, "Population", 1e-3)
    elif key == "medinc_units":
        w = unit_change(w, "MedInc", 10.0)
    elif key == "missing_spike":
        w = missing_spike(w, "AveOccup", 0.30, rng)
    w = with_predictions(w, model)
    if key == "concept_shift":
        w = label_shift(w, TARGET, 1.2)
    return w


def window_record(report: DriftReport, alpha: float = 0.05) -> dict:
    """Drift scores, error change and every detector's decision for one checked window."""
    feats = [c for c in report.columns if c.role == "feature"]
    pred = next(c for c in report.columns if c.role == "prediction")
    ks_p = np.array([1.0 if c.ks_p is None else c.ks_p for c in feats])
    max_psi = max(c.psi for c in feats)
    detectors = {
        "psi_gt_0.2": max_psi > 0.2,
        "psi_gt_0.1": max_psi > 0.1,
        "ks_uncorrected": bool((ks_p < alpha).any()),
        "ks_bh": bool((benjamini_hochberg(ks_p) <= alpha).any()),
        "psi_calibrated": any(c.drift for c in feats),
        "missing_rate": any(c.missing_drift for c in feats),
        "domain_classifier": report.domain_drift,
        "prediction_drift": pred.drift,
        "driftwatch_alarm": report.drift_detected,
        "pipeline_triage": bool(report.pipeline_issues),
        "label_drift": report.label_drift,
        "performance": report.performance_degraded,
    }
    return {
        "max_psi": max_psi,
        "domain_auc": report.domain.auc,
        "prediction_psi": pred.psi,
        "n_features_drifted": sum(c.drift or c.missing_drift for c in feats),
        "mae_ratio": report.performance.mae_ratio,
        "rmse_ratio": report.performance.rmse_ratio,
        "triage": report.triage.kind,
        **{k: bool(v) for k, v in detectors.items()},
    }


def run_scenarios(
    monitor: DriftMonitor,
    pools: dict[str, pd.DataFrame],
    model: HistGradientBoostingRegressor,
    houseage_value: float,
    n_windows: int,
    n_gradual: int,
    seed: int,
    window: int = WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All scenario windows and the gradual income-shift sequence, one row per window."""
    rows = []
    ref_median = float(monitor.reference["MedInc"].median())
    for s_i, sc in enumerate(SCENARIOS):
        rng = np.random.default_rng([seed, s_i])
        for w_i in range(n_windows):
            w = scenario_window(sc.key, pools, model, rng, houseage_value, n=window)
            rec = window_record(monitor.check(w, name=sc.key))
            rows.append({"scenario": sc.key, "group": sc.group, "window": w_i,
                         "median_income_ratio": float(w["MedInc"].median() / ref_median), **rec})
    grad = []
    rng = np.random.default_rng([seed, 99])
    for w_i, beta in enumerate(np.linspace(0.0, 1.0, n_gradual)):
        w = scenario_window("gradual", pools, model, rng, houseage_value, n=window,
                            beta=float(beta))
        rec = window_record(monitor.check(w, name=f"gradual-{w_i}"))
        grad.append({
            "scenario": "gradual_income", "group": "covariate", "window": w_i,
            "beta": float(beta), "median_income_ratio": float(w["MedInc"].median() / ref_median),
            **rec,
        })
    return pd.DataFrame(rows), pd.DataFrame(grad)


# --------------------------------------------------------------------------- PSI study


def psi_false_alarm_study(
    population: pd.DataFrame,
    n_reference: int,
    sizes: list[int],
    bins: list[int],
    reps: int,
    seed: int,
) -> dict:
    """No-drift simulation: reference and current windows drawn from the same population.

    For each replicate a reference sample (``n_reference`` rows) and current windows of every
    size are drawn with replacement from ``population``; PSI (every bin count) and KS are
    computed per feature with the same code paths the monitor uses.
    """
    rng = np.random.default_rng(seed)
    feats = list(FEATURES)
    values = {f: population[f].to_numpy(dtype=np.float64) for f in feats}
    n_pop = len(population)
    psi_vals = np.empty((len(sizes), len(bins), reps, len(feats)))
    ks_p = np.empty((len(sizes), reps, len(feats)))
    eff_bins = np.empty((len(bins), reps, len(feats)))
    for r in range(reps):
        ref_idx = rng.integers(0, n_pop, n_reference)
        cur_idx = [rng.integers(0, n_pop, n) for n in sizes]
        for j, f in enumerate(feats):
            ref = np.sort(values[f][ref_idx])
            ref_bins = [reference_bins(ref, b) for b in bins]
            for b_i, (edges, _) in enumerate(ref_bins):
                eff_bins[b_i, r, j] = edges.size + 1
            for s_i in range(len(sizes)):
                cur = np.sort(values[f][cur_idx[s_i]])
                for b_i, (edges, p_ref) in enumerate(ref_bins):
                    psi_vals[s_i, b_i, r, j] = psi_against(edges, p_ref, cur)
                ks_p[s_i, r, j] = ks_p_value(
                    ks_statistic_sorted(ref, cur), n_reference, sizes[s_i]
                )
    from scipy import special

    out: dict = {
        "n_reference": n_reference,
        "sizes": sizes,
        "bins": bins,
        "reps": reps,
        "per_feature": {},
        "any_feature": {},
        "threshold_95": {},
        "chi2_approx_far_0.2": {},
    }
    for b_i, b in enumerate(bins):
        far = psi_vals[:, b_i] > 0.2  # sizes x reps x feats
        out["per_feature"][str(b)] = {
            f: [float(x) for x in far[:, :, j].mean(axis=1)] for j, f in enumerate(feats)
        }
        out["per_feature"][str(b)]["mean_over_features"] = [
            float(x) for x in far.mean(axis=(1, 2))
        ]
        out["any_feature"][f"psi_gt_0.2_bins{b}"] = [float(x) for x in far.any(axis=2).mean(1)]
        out["any_feature"][f"psi_gt_0.1_bins{b}"] = [
            float(x) for x in (psi_vals[:, b_i] > 0.1).any(axis=2).mean(1)
        ]
        out["threshold_95"][str(b)] = {
            f: [float(np.quantile(psi_vals[s_i, b_i, :, j], 0.95)) for s_i in range(len(sizes))]
            for j, f in enumerate(feats)
        }
        k_eff = float(np.median(eff_bins[b_i, :, 0]))  # MedInc: continuous, so k_eff == b
        out["chi2_approx_far_0.2"][str(b)] = [
            float(special.chdtrc(k_eff - 1, 0.2 / (1 / n_reference + 1 / n))) for n in sizes
        ]
    out["any_feature"]["ks_uncorrected"] = [float(x) for x in (ks_p < 0.05).any(axis=2).mean(1)]
    out["any_feature"]["ks_bonferroni"] = [
        float(np.mean([(bonferroni(ks_p[s_i, r]) <= 0.05).any() for r in range(reps)]))
        for s_i in range(len(sizes))
    ]
    out["any_feature"]["ks_bh"] = [
        float(np.mean([(benjamini_hochberg(ks_p[s_i, r]) <= 0.05).any() for r in range(reps)]))
        for s_i in range(len(sizes))
    ]
    out["effective_bins_median"] = {
        str(b): {f: float(np.median(eff_bins[b_i, :, j])) for j, f in enumerate(feats)}
        for b_i, b in enumerate(bins)
    }
    return out


def calibrated_far_study(
    population: pd.DataFrame,
    feature: str,
    n_reference: int,
    sizes: list[int],
    n_bins: int,
    n_refs: int,
    n_boot: int,
    n_eval: int,
    alpha: float,
    seed: int,
) -> dict:
    """False-alarm rate of the bootstrap-calibrated PSI threshold, as used in production.

    For each of ``n_refs`` reference samples the threshold is calibrated from that reference
    alone (``n_boot`` bootstrap replicates); its false-alarm rate is then measured on
    ``n_eval`` fresh no-drift windows from the population.
    """
    rng = np.random.default_rng(seed)
    values = population[feature].to_numpy(dtype=np.float64)
    far = np.empty((len(sizes), n_refs))
    thresholds = np.empty((len(sizes), n_refs))
    for r in range(n_refs):
        ref = np.sort(values[rng.integers(0, values.size, n_reference)])
        edges, p_ref = reference_bins(ref, n_bins)
        for s_i, n in enumerate(sizes):
            null = psi_null_distribution(ref, n, n_bins, n_boot, seed=int(rng.integers(1 << 31)))
            t = float(np.quantile(null, 1 - alpha))
            thresholds[s_i, r] = t
            cur = values[rng.integers(0, values.size, (n_eval, n))]
            cur.sort(axis=1)
            psi_eval = np.array([psi_against(edges, p_ref, row) for row in cur])
            far[s_i, r] = np.mean(psi_eval > t)
    return {
        "feature": feature,
        "n_bins": n_bins,
        "alpha": alpha,
        "n_refs": n_refs,
        "n_boot": n_boot,
        "n_eval": n_eval,
        "sizes": sizes,
        "far_mean": [float(x) for x in far.mean(axis=1)],
        "far_p10": [float(x) for x in np.quantile(far, 0.1, axis=1)],
        "far_p90": [float(x) for x in np.quantile(far, 0.9, axis=1)],
        "threshold_median": [float(x) for x in np.median(thresholds, axis=1)],
    }


def pooled_cv_auc_test(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], n_splits: int, seed: int
) -> tuple[float, float]:
    """The naive design: pool out-of-fold scores from K-fold CV into one AUC, Mann-Whitney p."""
    X, y, _ = encode_frame(reference, current, columns)
    oof = np.empty(len(y))
    for train, test in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        clf = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20,
            early_stopping=False, random_state=seed,
        ).fit(X[train], y[train])
        oof[test] = clf.predict_proba(X[test])[:, 1]
    auc = float(roc_auc_score(y, oof))
    return auc, auc_null_p_value(auc, len(reference), len(current))


def domain_classifier_calibration(
    population: pd.DataFrame, n_reference: int, n_current: int, trials: int, seed: int
) -> dict:
    """False-alarm rate of two domain-classifier tests when there is no drift at all.

    Each trial splits the population into disjoint random reference and current samples.
    """
    rng = np.random.default_rng(seed)
    cols = list(FEATURES)
    pooled_auc, pooled_p, cf_auc, cf_p, z = [], [], [], [], []
    for t in range(trials):
        idx = rng.permutation(len(population))
        ref = population.iloc[idx[:n_reference]]
        cur = population.iloc[idx[n_reference : n_reference + n_current]]
        a, p = pooled_cv_auc_test(ref, cur, cols, 5, seed=t)
        pooled_auc.append(a)
        pooled_p.append(p)
        res = domain_classifier_test(ref, cur, cols, importance=False, seed=t)
        cf_auc.append(res.auc)
        cf_p.append(res.p_value)
        n1 = n_reference / res.n_splits
        n0 = n_current / res.n_splits
        se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
        z.append([(fa - 0.5) / se for fa in res.fold_auc])
    se_full = np.sqrt((n_reference + n_current + 1) / (12 * n_reference * n_current))
    z = np.array(z)
    return {
        "trials": trials,
        "n_reference": n_reference,
        "n_current": n_current,
        "pooled_5fold": {
            "false_alarm_rate": float(np.mean(np.array(pooled_p) < 0.05)),
            "auc_mean": float(np.mean(pooled_auc)),
            "auc_sd": float(np.std(pooled_auc, ddof=1)),
            "mann_whitney_sd": float(se_full),
        },
        "crossfit_2fold_bonferroni": {
            "false_alarm_rate": float(np.mean(np.array(cf_p) < 0.05)),
            "auc_mean": float(np.mean(cf_auc)),
            "fold_z_correlation": float(np.corrcoef(z.T)[0, 1]),
            "fold_z_sd": [float(x) for x in z.std(axis=0, ddof=1)],
        },
    }


# --------------------------------------------------------------------------- summaries


def column_table(report: DriftReport) -> list[dict]:
    rows = list(report.columns) + ([report.label] if report.label is not None else [])
    return [
        {
            "column": c.column, "role": c.role, "psi": c.psi, "ks": c.ks, "ks_p": c.ks_p,
            "wasserstein_norm": c.wasserstein, "p_value": c.p_value, "p_adjusted": c.p_adjusted,
            "drift": c.drift,
        }
        for c in rows
    ]


def summarise_scenarios(windows: pd.DataFrame) -> dict:
    det_keys = [k for k, _ in DETECTORS]
    out = {}
    for sc in SCENARIOS:
        w = windows[windows["scenario"] == sc.key]
        out[sc.key] = {
            "label": sc.label.replace("\n", " "),
            "group": sc.group,
            "description": sc.description,
            "windows": int(len(w)),
            "mae_ratio_mean": float(w["mae_ratio"].mean()),
            "mae_ratio_sd": float(w["mae_ratio"].std()),
            "median_income_ratio_mean": float(w["median_income_ratio"].mean()),
            "domain_auc_mean": float(w["domain_auc"].mean()),
            "max_psi_mean": float(w["max_psi"].mean()),
            "prediction_psi_mean": float(w["prediction_psi"].mean()),
            "alarm_rate": {k: float(w[k].mean()) for k in det_keys},
            "triage": {k: float(v) for k, v in w["triage"].value_counts(normalize=True)
                       .sort_index().items()},
        }
    return out


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank().corr(y.rank()))


def drift_vs_performance(windows: pd.DataFrame, gradual: pd.DataFrame) -> dict:
    allw = pd.concat([windows, gradual], ignore_index=True)
    scores = ["domain_auc", "max_psi", "prediction_psi", "n_features_drifted"]
    subsets = {
        "all_windows": allw,
        "excluding_concept_shift": allw[allw["scenario"] != "concept_shift"],
        "covariate_shift_only": allw[allw["group"].isin(["none", "real", "covariate"])],
        "pipeline_faults_only": allw[allw["group"].isin(["none", "pipeline"])],
    }
    out = {
        name: {"n": int(len(d)), **{s: spearman(d[s], d["mae_ratio"]) for s in scores}}
        for name, d in subsets.items()
    }
    # How often is an input-drift alarm accompanied by real damage, and vice versa?
    damaged = allw["mae_ratio"] > 1.10
    alarm = allw["driftwatch_alarm"]
    out["confusion_vs_mae_gt_1.1"] = {
        "alarm_and_damage": int((alarm & damaged).sum()),
        "alarm_no_damage": int((alarm & ~damaged).sum()),
        "no_alarm_damage": int((~alarm & damaged).sum()),
        "no_alarm_no_damage": int((~alarm & ~damaged).sum()),
        "alarm_no_damage_by_scenario": {
            k: int(v) for k, v in allw[alarm & ~damaged]["scenario"].value_counts()
            .sort_index().items()
        },
        "no_alarm_damage_by_scenario": {
            k: int(v) for k, v in allw[~alarm & damaged]["scenario"].value_counts()
            .sort_index().items()
        },
    }
    return out


def first_alarm(gradual: pd.DataFrame) -> dict:
    out = {}
    for key, _ in DETECTORS:
        hit = gradual.index[gradual[key]].tolist()
        if hit:
            i = hit[0]
            out[key] = {
                "window": int(gradual.loc[i, "window"]),
                "beta": float(gradual.loc[i, "beta"]),
                "median_income_ratio": float(gradual.loc[i, "median_income_ratio"]),
                "mae_ratio": float(gradual.loc[i, "mae_ratio"]),
                "alarm_share_after_first": float(gradual.loc[i:, key].mean()),
            }
        else:
            out[key] = None
    return out


# --------------------------------------------------------------------------- main


def run(
    out_dir: Path,
    data_home: Path,
    n_windows: int = 40,
    n_gradual: int = 30,
    psi_reps: int = 600,
    calib_refs: int = 40,
    dc_trials: int = 150,
    scale: float = 1.0,
    window: int = WINDOW,
    df: pd.DataFrame | None = None,
) -> dict:
    """Run the whole study; write metrics.json, the figures and the CLI demo files.

    ``scale`` < 1 shrinks every replication count (for smoke tests); ``df`` replaces the
    downloaded California Housing frame (same columns), e.g. with synthetic data in tests.
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    demo_dir = data_home / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_california(data_home)
    n_boot = max(100, int(1000 * scale))
    splits = split_regions(df, seed=0)
    model = train_model(splits.train)
    ref = with_predictions(splits.reference, model)
    hold = with_predictions(splits.holdout, model)
    north = with_predictions(splits.north, model)
    model_metrics = {
        "type": "HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)",
        "features": list(FEATURES),
        "mae_reference": mae(ref[TARGET], ref[PREDICTION]),
        "mae_holdout": mae(hold[TARGET], hold[PREDICTION]),
        "mae_north": mae(north[TARGET], north[PREDICTION]),
        "rmse_reference": rmse(ref[TARGET], ref[PREDICTION]),
        "rmse_north": rmse(north[TARGET], north[PREDICTION]),
        "r2_reference": float(
            1 - np.sum((ref[TARGET] - ref[PREDICTION]) ** 2)
            / np.sum((ref[TARGET] - ref[TARGET].mean()) ** 2)
        ),
        "mean_label_reference": float(ref[TARGET].mean()),
        "mean_label_north": float(north[TARGET].mean()),
        "permutation_importance_mae": permutation_importance_mae(model, ref),
        "population_units_share_below_train_min": float(
            (splits.holdout["Population"] / 1000 < splits.train["Population"].min()).mean()
        ),
    }

    # (1) the real geographic shift, full North/Central sample vs the reference
    fit_kw = {"features": list(FEATURES), "target": TARGET, "prediction": PREDICTION}
    monitor = DriftMonitor(DriftConfig(n_bootstrap=n_boot)).fit(
        ref, name="Southern CA (reference)", **fit_kw
    )
    north_shuffled = north.sample(frac=1.0, random_state=0).reset_index(drop=True)
    north_report = monitor.check(north_shuffled, name="Northern/Central CA")
    rule = DriftMonitor(DriftConfig(psi_threshold=0.2, domain_classifier=False)).fit(ref, **fit_kw)
    rule_report = rule.check(north_shuffled)

    # demo files for the CLI (written to data/, which is git-ignored)
    ref.to_parquet(demo_dir / "reference.parquet", index=False)
    north_shuffled.to_parquet(demo_dir / "north.parquet", index=False)
    hold.to_parquet(demo_dir / "holdout.parquet", index=False)

    # (2) PSI false alarms under no drift
    south = df[df["Latitude"] < 35.0].reset_index(drop=True)
    sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    psi_study = psi_false_alarm_study(
        south, len(ref), sizes, [5, 10, 20], max(50, int(psi_reps * scale)), SEED + 1
    )
    calib = calibrated_far_study(
        south, "MedInc", len(ref), sizes, 10, max(5, int(calib_refs * scale)),
        n_boot=max(50, int(200 * scale)), n_eval=200, alpha=0.05, seed=SEED + 2,
    )

    dc_calib = domain_classifier_calibration(
        south, len(ref), window, max(10, int(dc_trials * scale)), SEED + 4
    )

    # (3) scenario windows
    win_monitor = DriftMonitor(DriftConfig(domain_importance=False, n_bootstrap=n_boot)).fit(
        ref, **fit_kw
    )
    windows, gradual = run_scenarios(
        win_monitor,
        {"holdout": splits.holdout, "north": splits.north},
        model,
        houseage_value=float(splits.reference["HouseAge"].median()),
        n_windows=max(3, int(n_windows * scale)),
        n_gradual=max(4, int(n_gradual * scale)),
        seed=SEED + 3,
        window=window,
    )

    metrics = {
        "driftwatch_version": __version__,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "data": {
            "source": "California Housing (1990 US census), sklearn.datasets",
            "rows": int(len(df)), "south_rows": int(len(south)),
            "north_rows": int(len(splits.north)), "train_rows": int(len(splits.train)),
            "reference_rows": int(len(splits.reference)),
            "holdout_rows": int(len(splits.holdout)), "latitude_split": 35.0,
        },
        "model": model_metrics,
        "north_vs_south": {
            "columns": column_table(north_report),
            "domain_auc": north_report.domain.auc,
            "domain_importances": north_report.domain.importances,
            "mae_ratio": north_report.performance.mae_ratio,
            "rmse_ratio": north_report.performance.rmse_ratio,
            "performance_p_value": north_report.performance.p_value,
            "max_feature_psi": max(c.psi for c in north_report.columns if c.role == "feature"),
            "features_drifted_driftwatch": sum(
                c.drift for c in north_report.columns if c.role == "feature"),
            "features_over_psi_0.2": [c.column for c in rule_report.columns
                                      if c.role == "feature" and c.drift],
            "alarms": north_report.alarms(),
        },
        "psi_false_alarm": psi_study,
        "psi_calibrated_false_alarm": calib,
        "domain_classifier_calibration": dc_calib,
        "scenarios": {
            "window_rows": window,
            "windows_per_scenario": int(windows.groupby("scenario").size().iloc[0]),
            "detectors": dict(DETECTORS),
            "results": summarise_scenarios(windows),
        },
        "gradual_income_shift": {
            "windows": int(len(gradual)),
            "mae_ratio_max": float(gradual["mae_ratio"].max()),
            "performance_alarms": int(gradual["performance"].sum()),
            "beta_range": [0.0, 1.0],
            "median_income_ratio_last": float(gradual["median_income_ratio"].iloc[-1]),
            "mae_ratio_last": float(gradual["mae_ratio"].iloc[-1]),
            "first_alarm": first_alarm(gradual),
        },
        "drift_vs_performance": drift_vs_performance(windows, gradual),
    }
    make_figures(metrics, north_report, windows, gradual, out_dir / "figures")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"study finished in {time.time() - t0:.1f} s")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--scale", type=float, default=1.0,
                        help="shrink replication counts (e.g. 0.1 for a quick smoke run)")
    args = parser.parse_args()
    m = run(args.out, args.data, scale=args.scale)
    print(json.dumps({"north_mae_ratio": m["north_vs_south"]["mae_ratio"],
                      "north_domain_auc": m["north_vs_south"]["domain_auc"]}, indent=2))


if __name__ == "__main__":
    main()
