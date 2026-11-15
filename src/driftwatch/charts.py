"""Matplotlib charts shared by the HTML report and the README figures."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 9.5,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
}


def png_bytes(fig: plt.Figure, dpi: int = 110) -> bytes:
    """Render a figure to PNG bytes (no timestamps, so output is deterministic)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _plot_range(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """x-range covering the bulk of both samples (0.5-99.5th percentiles), robust to outliers."""
    lo = min(np.quantile(ref, 0.005), np.quantile(cur, 0.005))
    hi = max(np.quantile(ref, 0.995), np.quantile(cur, 0.995))
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    return float(lo), float(hi)


def distribution_panel(
    ax: plt.Axes,
    reference: pd.Series,
    current: pd.Series,
    kind: str,
    title: str,
    drifted: bool,
    labels: tuple[str, str] = ("reference", "current"),
) -> None:
    """Overlay reference and current distributions of one column on ``ax``."""
    if kind == "numeric":
        ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
        cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
        if ref.size and cur.size:
            lo, hi = _plot_range(ref, cur)
            integer = np.all(ref == np.round(ref)) and np.all(cur == np.round(cur))
            if integer and hi - lo <= 200:  # integer-valued: align bin edges with the integers
                step = max(1.0, float(np.ceil((hi - lo + 1) / 40)))
                bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + step + 0.5, step)
            else:
                bins = np.linspace(lo, hi, 36)
            kw = {"bins": bins, "density": True, "histtype": "stepfilled", "linewidth": 0}
            ax.hist(np.clip(ref, lo, hi), color=BLUE, alpha=0.30, **kw)
            ax.hist(np.clip(cur, lo, hi), color=ORANGE, alpha=0.30, **kw)
            kw.update(histtype="step", linewidth=1.6)
            ax.hist(np.clip(ref, lo, hi), color=BLUE, label=labels[0], **kw)
            ax.hist(np.clip(cur, lo, hi), color=ORANGE, label=labels[1], **kw)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    else:
        r = reference.dropna().astype(str).value_counts(normalize=True)
        c = current.dropna().astype(str).value_counts(normalize=True)
        cats = list((r.add(c, fill_value=0)).sort_values(ascending=False).index[:8])
        x = np.arange(len(cats))
        ax.bar(x - 0.2, r.reindex(cats, fill_value=0), 0.38, color=BLUE, label=labels[0])
        ax.bar(x + 0.2, c.reindex(cats, fill_value=0), 0.38, color=ORANGE, label=labels[1])
        ax.set_xticks(x, cats, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="x", visible=False)
    ax.set_title(title, fontsize=9.5, color=CRITICAL if drifted else INK)
    ax.spines["left"].set_visible(False)


def distribution_grid(report: object, ncols: int = 4) -> bytes:
    """Small multiples of every monitored column, reference vs current (PNG bytes)."""
    cols = list(report.columns) + ([report.label] if report.label is not None else [])
    ref_df, cur_df = report.reference_data, report.current_data
    nrows = int(np.ceil(len(cols) / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.25 * nrows), squeeze=False)
        for ax, c in zip(axes.ravel(), cols, strict=False):
            flag = c.drift or c.missing_drift
            title = f"{c.column}  PSI {c.psi:.2f}" + ("  DRIFT" if flag else "")
            distribution_panel(ax, ref_df[c.column], cur_df[c.column], c.kind, title, flag)
        for ax in axes.ravel()[len(cols):]:
            ax.set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.04))
        fig.tight_layout()
        return png_bytes(fig)


def importance_chart(importances: dict[str, float], auc: float) -> bytes:
    """Horizontal bars: drop in domain-classifier AUC when each column is permuted."""
    items = sorted(importances.items(), key=lambda kv: kv[1])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.0, 0.32 * len(items) + 1.0))
        ax.barh([k for k, _ in items], [max(v, 0) for _, v in items], color=BLUE, height=0.6)
        ax.set_xlabel("AUC drop when the column is shuffled")
        ax.set_title(f"What separates the samples (domain-classifier AUC = {auc:.3f})")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        return png_bytes(fig)


def rolling_chart(rolling: pd.DataFrame, mae_reference: float, max_ratio: float) -> bytes:
    """Rolling MAE of the current window against the reference MAE and the tolerance."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        ax.plot(rolling.index, rolling["mae"], color=ORANGE, lw=1.6, label="rolling MAE (current)")
        ax.axhline(mae_reference, color=BLUE, lw=1.4, label="reference MAE")
        ax.axhline(mae_reference * max_ratio, color=INK2, lw=1.0, ls="--",
                   label=f"tolerance ({max_ratio:.2f}x)")
        ax.set_xlabel("row of the current data (time order)")
        ax.set_ylabel("MAE")
        ax.set_title("Rolling error once labels arrive")
        ax.legend(loc="upper left", ncol=3)
        fig.tight_layout()
        return png_bytes(fig)
