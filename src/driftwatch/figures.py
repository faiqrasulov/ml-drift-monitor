"""The four README figures (static PNG, dpi 150)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from driftwatch.charts import (
    AQUA,
    BLUE,
    GRID,
    INK,
    INK2,
    MUTED,
    ORANGE,
    STYLE,
    SURFACE,
    distribution_panel,
)
from driftwatch.monitor import DriftReport
from driftwatch.scenarios import DETECTORS, SCENARIOS

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _pct(ax: plt.Axes, axis: str = "y") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _p(p: float | None) -> str:
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------- figure 1


def fig_north_dashboard(m: dict, report: DriftReport, path: Path) -> None:
    """Per-feature drift dashboard for Northern/Central vs Southern California."""
    ns = m["north_vs_south"]
    cols = [c for c in report.columns]
    ref, cur = report.reference_data, report.current_data
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4))
        for ax, c in zip(axes.ravel(), cols, strict=False):
            name = c.column if c.role == "feature" else "model prediction"
            distribution_panel(ax, ref[c.column], cur[c.column], c.kind, name, False,
                               labels=("Southern CA (reference)", "Northern/Central CA"))
            flag = "drift" if c.drift else "no drift"
            txt = (f"PSI {c.psi:.3f}\nKS D {c.ks:.3f}\nW/sd {c.wasserstein:.2f}\n"
                   f"p adj {_p(c.p_adjusted)}: {flag}")
            ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color=INK2, linespacing=1.35,
                    bbox={"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": GRID, "lw": 0.6})
        ax = axes.ravel()[len(cols)]
        imp = sorted(ns["domain_importances"].items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in imp], [max(v, 0.0) for _, v in imp], color=BLUE, height=0.6)
        ax.set_title(f"Domain classifier, AUC {ns['domain_auc']:.3f}", fontsize=9.5)
        ax.set_xlabel("AUC drop when feature is shuffled")
        ax.grid(axis="y", visible=False)
        for a in axes.ravel()[len(cols) + 1:]:
            a.set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 0.97))
        over = ns["features_over_psi_0.2"]
        fig.suptitle(
            "Real geographic shift: every feature drifts significantly, "
            f"{'none' if not over else len(over)} crosses PSI 0.2, and model MAE rises "
            f"{ns['mae_ratio'] - 1:.0%}",
            x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.035,
        )
        fig.text(
            0.01, 0.982,
            f"Reference: {report.n_reference:,} Southern California block groups (held out from "
            f"training). Current: {report.n_current:,} Northern/Central block groups. "
            "p adj = bootstrap-calibrated PSI p-value, Benjamini-Hochberg adjusted.",
            ha="left", fontsize=9, color=INK2,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        _save(fig, path)


# --------------------------------------------------------------------------- figure 2


def fig_psi_false_alarms(m: dict, path: Path) -> None:
    """False-alarm rate of PSI > 0.2 and the calibrated threshold vs window size."""
    st = m["psi_false_alarm"]
    cal = m["psi_calibrated_false_alarm"]
    sizes = np.array(st["sizes"])
    colors = {"5": AQUA, "10": BLUE, "20": ORANGE}
    with plt.rc_context(STYLE):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.7))
        for b, color in colors.items():
            far = st["per_feature"][b]["MedInc"]
            ax.plot(sizes, far, "o-", color=color, label=f"PSI > 0.2, {b} bins")
            ax.plot(sizes, st["chi2_approx_far_0.2"][b], ":", color=color, lw=1.3)
        ax.plot(sizes, cal["far_mean"], "s-", color=INK, lw=1.6, ms=5,
                label="bootstrap-calibrated threshold, 10 bins")
        ax.fill_between(sizes, cal["far_p10"], cal["far_p90"], color=INK, alpha=0.08, lw=0,
                        label="calibrated: 10th-90th pct over reference samples")
        ax.axhline(0.05, color=INK2, lw=1, ls="--", label="target false-alarm rate 5%")
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
        ax.minorticks_off()
        ax.set_ylim(-0.02, max(0.6, max(st["per_feature"]["20"]["MedInc"]) + 0.05))
        _pct(ax)
        ax.set_xlabel("current window size (rows)")
        ax.set_ylabel("false-alarm rate (no drift)")
        ax.set_title("'PSI > 0.2' false alarms depend on window size and bins")
        ax.legend(loc="upper right")
        ax.text(0.99, 0.5, "dotted: chi-square approximation", transform=ax.transAxes,
                ha="right", fontsize=8, color=MUTED)

        for b, color in colors.items():
            ax2.plot(sizes, st["threshold_95"][b]["MedInc"], "o-", color=color,
                     label=f"{b} bins")
        for level in (0.2, 0.1):
            ax2.axhline(level, color=INK2, lw=1, ls="--")
            ax2.text(sizes[0], level * 1.08, f"rule of thumb {level}", color=INK2, fontsize=8.5)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xticks(sizes, [f"{s:,}" for s in sizes])
        ax2.minorticks_off()
        ax2.set_xlabel("current window size (rows)")
        ax2.set_ylabel("PSI exceeded 5% of the time under no drift")
        ax2.set_title("The threshold that gives 5% false alarms")
        ax2.legend(loc="upper right")
        fig.suptitle(
            f"No-drift simulation on MedInc (Southern California, reference n = "
            f"{st['n_reference']:,}, {st['reps']:,} replicates per point)",
            x=0.01, ha="left", fontsize=9.5, color=INK2, y=0.995, fontweight="normal",
        )
        fig.tight_layout()
        _save(fig, path)


# --------------------------------------------------------------------------- figure 3


def fig_detector_heatmap(m: dict, path: Path) -> None:
    """Alarm rate of every detector in every scenario (windows of 500 rows)."""
    res = m["scenarios"]["results"]
    mat = np.array([[res[s.key]["alarm_rate"][d] for s in SCENARIOS] for d, _ in DETECTORS])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13.0, 7.0))
        ax.imshow(mat, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.55 else INK)
        xl = [f"{s.label}\nMAE x{res[s.key]['mae_ratio_mean']:.2f}" for s in SCENARIOS]
        ax.set_xticks(range(len(SCENARIOS)), xl, fontsize=8.3)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(DETECTORS)), [label for _, label in DETECTORS], fontsize=8.8)
        for y in (1.5, 3.5, 9.5):
            ax.axhline(y, color=SURFACE, lw=3)
        for x in (0.5, 1.5, 3.5, 7.5):
            ax.axvline(x, color=SURFACE, lw=3)
        ax.tick_params(length=0)
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(False)
        n = m["scenarios"]["windows_per_scenario"]
        ax.set_title(
            f"Share of {n} windows ({m['scenarios']['window_rows']} rows each) in which each "
            "detector fires.  Column 1 = false alarms; MAE x = mean error ratio vs reference",
            fontsize=9.5, color=INK2, fontweight="normal", pad=58, loc="left",
        )
        fig.suptitle("Which detector fires in which scenario", x=0.01, ha="left",
                     fontsize=12.5, fontweight="bold", y=1.0)
        fig.tight_layout()
        _save(fig, path)


# --------------------------------------------------------------------------- figure 4

GROUP_STYLE = {
    "none": ("No drift (same region)", MUTED, "o"),
    "real": ("North/Central CA (real shift)", BLUE, "D"),
    "covariate": ("Income shift (injected, incl. gradual)", BLUE, "o"),
    "pipeline": ("Pipeline faults (injected)", ORANGE, "s"),
    "concept": ("Prices +20% (concept shift)", AQUA, "^"),
}

SHORT = {
    "no_drift": "no drift", "north": "North/Central", "income_mild": "income mild",
    "income_strong": "income strong", "houseage_stuck": "HouseAge stuck",
    "population_units": "Population /1000", "medinc_units": "MedInc x10",
    "missing_spike": "AveOccup 30% missing", "concept_shift": "prices +20%",
}

# Label offsets in points, per panel, chosen so the scenario labels do not collide.
LABEL_OFFSETS = {
    "domain_auc": {
        "no_drift": (-8, -16), "north": (6, 6), "income_mild": (4, 12),
        "income_strong": (6, -14), "houseage_stuck": (-10, 8), "population_units": (-10, 8),
        "medinc_units": (-10, -4), "missing_spike": (-40, 16), "concept_shift": (6, 4),
    },
    "max_psi": {
        "no_drift": (-30, -16), "north": (6, 6), "income_mild": (-10, 14),
        "income_strong": (4, -16), "houseage_stuck": (-10, 8), "population_units": (-10, 8),
        "medinc_units": (-10, -4), "missing_spike": (-70, 20), "concept_shift": (6, 4),
    },
}


def fig_drift_vs_performance(m: dict, windows: pd.DataFrame, gradual: pd.DataFrame,
                             path: Path) -> None:
    """Drift score vs change in MAE, one point per window."""
    allw = pd.concat([windows, gradual], ignore_index=True)
    dv = m["drift_vs_performance"]["all_windows"]
    dvx = m["drift_vs_performance"]["excluding_concept_shift"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
        specs = [("domain_auc", "domain-classifier AUC (multivariate drift score)", False),
                 ("max_psi", "largest feature PSI (log scale)", True)]
        for ax, (col, xlabel, logx) in zip(axes, specs, strict=True):
            for g, (label, color, marker) in GROUP_STYLE.items():
                d = allw[allw["group"] == g]
                ax.scatter(d[col], d["mae_ratio"], s=18, color=color, marker=marker,
                           alpha=0.45, lw=0, label=label)
            for key, short in SHORT.items():
                d = windows[windows["scenario"] == key]
                x, y = d[col].median(), d["mae_ratio"].median()
                ha = "right" if LABEL_OFFSETS[col][key][0] < 0 else "left"
                ax.annotate(short, (x, y), xytext=LABEL_OFFSETS[col][key],
                            textcoords="offset points", fontsize=8, color=INK, ha=ha,
                            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
            ax.axhline(1.0, color=INK2, lw=0.9)
            ax.axhline(1.1, color=INK2, lw=0.9, ls="--")
            ax.set_yscale("log")
            ticks = [0.9, 1.0, 1.1, 1.25, 1.5, 2, 3, 5]
            ax.set_yticks(ticks, [f"{t:g}x" for t in ticks])
            ax.minorticks_off()
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_title(f"Spearman rho = {dv[col]:.2f} (all windows), "
                         f"{dvx[col]:.2f} without concept shift", fontsize=9.8)
        axes[0].set_ylabel("MAE / reference MAE (log scale)")
        axes[1].text(1.0, 1.112, "tolerance 1.1x ", transform=axes[1].get_yaxis_transform(),
                     fontsize=8, color=INK2, va="bottom", ha="right")
        axes[0].legend(loc="upper left", fontsize=8)
        fig.suptitle("Drift is not damage: drift scores vs the actual change in model error "
                     f"({len(allw)} windows of {m['scenarios']['window_rows']} rows)",
                     x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.0)
        fig.tight_layout()
        _save(fig, path)


def make_figures(m: dict, north_report: DriftReport, windows: pd.DataFrame,
                 gradual: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_north_dashboard(m, north_report, out / "north_vs_south_dashboard.png")
    fig_psi_false_alarms(m, out / "psi_false_alarms.png")
    fig_detector_heatmap(m, out / "detector_heatmap.png")
    fig_drift_vs_performance(m, windows, gradual, out / "drift_vs_performance.png")
