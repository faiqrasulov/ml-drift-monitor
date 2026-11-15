"""Self-contained HTML drift report: inline CSS, charts embedded as base64 PNG, no templates."""

from __future__ import annotations

from html import escape

from driftwatch import __version__
from driftwatch.charts import data_uri, distribution_grid, importance_chart, rolling_chart
from driftwatch.performance import rolling_metrics

CSS = """
:root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --rule:#e1e0d9;
  --blue:#2a78d6; --critical:#d03b3b; --good:#0a7d0a; --serious:#b4501f; --panel:#f4f3ef; }
* { box-sizing:border-box; }
body { margin:0; background:var(--surface); color:var(--ink);
  font:15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width:1120px; margin:0 auto; padding:32px 28px 48px; }
.eyebrow { margin:0; color:var(--muted); font-size:12px; letter-spacing:.08em;
  text-transform:uppercase; }
h1 { margin:4px 0 2px; font-size:26px; }
h2 { margin:36px 0 10px; font-size:19px; border-bottom:1px solid var(--rule); padding-bottom:6px; }
.sub { margin:0; color:var(--ink2); }
.verdict { margin:22px 0 0; padding:16px 20px; border-radius:10px; border:1px solid var(--rule);
  background:var(--panel); border-left:6px solid var(--good); }
.verdict.alarm { border-left-color:var(--critical); }
.verdict strong { font-size:18px; }
.verdict .headline { margin:2px 0 6px; font-size:20px; font-weight:650; }
.verdict .label { margin:10px 0 0; color:var(--ink2); font-size:12.5px; font-weight:600;
  text-transform:uppercase; letter-spacing:.05em; }
.verdict ul, .verdict ol { margin:4px 0 0; padding-left:22px; }
.verdict details { margin-top:10px; color:var(--ink2); font-size:14px; }
.verdict summary { cursor:pointer; font-weight:600; }
.kpis { display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-top:16px; }
.kpi { padding:12px 14px; border:1px solid var(--rule); border-radius:10px; }
.kpi .label { color:var(--ink2); font-size:12.5px; }
.kpi .value { font-size:24px; font-weight:650; margin-top:2px; }
.kpi .detail { color:var(--muted); font-size:12.5px; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th { text-align:left; color:var(--ink2); font-weight:600; border-bottom:1px solid var(--ink2);
  padding:6px 8px; }
td { border-bottom:1px solid var(--rule); padding:6px 8px; font-variant-numeric:tabular-nums; }
td.num, th.num { text-align:right; }
.tag { display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px;
  font-weight:600; border:1px solid currentColor; }
.tag.drift { color:var(--critical); } .tag.ok { color:var(--good); }
.tag.warn { color:var(--serious); }
img { max-width:100%; height:auto; display:block; margin:8px 0; }
.note { color:var(--ink2); font-size:13.5px; }
footer { margin-top:40px; color:var(--muted); font-size:12.5px; }
code { background:var(--panel); padding:1px 5px; border-radius:4px; font-size:12.5px; }
@media (max-width: 760px) { .kpis { grid-template-columns:repeat(2, 1fr); } }
"""


def _p(p: float | None) -> str:
    if p is None:
        return "&ndash;"
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


def _f(x: float | None, digits: int = 3) -> str:
    return "&ndash;" if x is None else f"{x:.{digits}f}"


def _status(drift: bool, missing: bool) -> str:
    if drift:
        return '<span class="tag drift">DRIFT</span>'
    if missing:
        return '<span class="tag warn">MISSING&nbsp;RATE</span>'
    return '<span class="tag ok">ok</span>'


def _kpi(label: str, value: str, detail: str) -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="detail">{detail}</div></div>'
    )


def render_html(report: object, title: str | None = None) -> str:
    """Render a :class:`~driftwatch.monitor.DriftReport` as one self-contained HTML page."""
    cfg = report.config
    title = title or f"Drift report: {report.current_name} vs {report.reference_name}"
    parts: list[str] = []

    # ---- verdict
    alarms = report.alarms()
    if alarms:
        tri = report.triage
        steps = "".join(f"<li>{escape(a)}</li>" for a in tri.actions)
        faults = "".join(f"<li>{escape(i)}</li>" for i in report.pipeline_issues)
        items = "".join(f"<li>{escape(a)}</li>" for a in alarms)
        verdict = (
            '<section class="verdict alarm"><p class="eyebrow">Alarm &middot; likely cause</p>'
            f'<p class="headline">{escape(tri.headline)}</p>'
            + (f'<p class="label">Pipeline-fault signatures</p><ul>{faults}</ul>' if faults else "")
            + f'<p class="label">What to do</p><ol>{steps}</ol>'
            f"<details><summary>{len(alarms)} check(s) fired</summary><ul>{items}</ul>"
            "</details></section>"
        )
    else:
        verdict = (
            '<section class="verdict"><strong>NO ALARM</strong><p class="note">No column, '
            "missing-rate, domain-classifier, label or performance check fired at "
            f"alpha = {cfg.alpha:g} after {escape(cfg.correction)} correction.</p></section>"
        )
    parts.append(verdict)

    # ---- KPI cards
    n_feat = sum(1 for c in report.columns if c.role == "feature")
    n_drift = sum(1 for c in report.columns if c.role == "feature" and (c.drift or c.missing_drift))
    cards = [_kpi("Features drifted", f"{n_drift} / {n_feat}",
                  f"{escape(report.columns[0].test) if report.columns else ''}, "
                  f"{escape(cfg.correction)} correction")]
    if report.domain is not None:
        d = report.domain
        detail = f"0.5 = indistinguishable; p<sub>adj</sub> {_p(report.domain_p_adjusted)}"
        cards.append(_kpi("Domain-classifier AUC", f"{d.auc:.3f}", detail))
    else:
        cards.append(_kpi("Domain-classifier AUC", "&ndash;", "disabled"))
    pred = [c for c in report.columns if c.role == "prediction"]
    if pred:
        cards.append(_kpi("Prediction drift (PSI)", f"{pred[0].psi:.3f}",
                          "drift" if pred[0].drift else "no significant drift"))
    else:
        cards.append(_kpi("Prediction drift (PSI)", "&ndash;", "no prediction column"))
    if report.performance is not None:
        pf = report.performance
        cards.append(_kpi("MAE vs reference", f"{pf.mae_ratio:.2f}x",
                          f"{pf.mae_current:.3f} vs {pf.mae_reference:.3f}; "
                          + ("degraded" if pf.degraded else "within tolerance")))
    else:
        cards.append(_kpi("MAE vs reference", "&ndash;", "labels or predictions not available"))
    parts.append(f'<section class="kpis">{"".join(cards)}</section>')

    # ---- per-column table
    rows = list(report.columns) + ([report.label] if report.label is not None else [])
    rows.sort(key=lambda c: (not (c.drift or c.missing_drift), -(c.psi if c.psi == c.psi else 0)))
    body = []
    for c in rows:
        dist = _f(c.wasserstein) if c.kind == "numeric" else _f(c.js)
        body.append(
            f"<tr><td><b>{escape(c.column)}</b></td><td>{c.role}</td><td>{c.kind}</td>"
            f'<td class="num">{_f(c.psi)}</td><td class="num">{_f(c.ks)}</td>'
            f'<td class="num">{dist}</td>'
            f'<td class="num">{c.missing_reference:.1%} &rarr; {c.missing_current:.1%}</td>'
            f"<td>{escape(c.test)}</td>"
            f'<td class="num">{_p(c.p_value)}</td><td class="num">{_p(c.p_adjusted)}</td>'
            f"<td>{_status(c.drift, c.missing_drift)}</td></tr>"
        )
    parts.append(
        "<h2>Per-column drift</h2><table><thead><tr><th>Column</th><th>Role</th><th>Type</th>"
        '<th class="num">PSI</th><th class="num">KS D</th>'
        '<th class="num">Wasserstein / JS</th><th class="num">Missing (ref &rarr; cur)</th>'
        '<th>Decision test</th><th class="num">p</th><th class="num">p adj.</th>'
        f"<th>Status</th></tr></thead><tbody>{''.join(body)}</tbody></table>"
        '<p class="note">Wasserstein distance is divided by the reference standard deviation '
        "(numeric columns); JS is the Jensen-Shannon distance in [0, 1] (categorical columns). "
        "PSI p-values come from a bootstrap of the reference at the current window size, so "
        "they already account for sample size and bin count. The label row (if any) is tested "
        "on its own and is not part of the input-drift correction family.</p>"
    )

    # ---- distributions
    if report.reference_data is not None and report.current_data is not None:
        img = data_uri(distribution_grid(report))
        parts.append(
            "<h2>Distributions</h2>"
            f'<img src="{img}" alt="Reference vs current distribution of every column">'
            '<p class="note">Blue: reference. Orange: current. Axes cover the 0.5-99.5th '
            "percentiles; values beyond are clipped into the outer bins.</p>"
        )

    # ---- domain classifier
    if report.domain is not None:
        d = report.domain
        parts.append(
            "<h2>Multivariate drift: domain classifier</h2>"
            f'<p class="note">A gradient-boosting classifier was trained to tell {d.n_reference:,} '
            f"reference rows from {d.n_current:,} current rows, cross-fitted over {d.n_splits} "
            "folds (each fold is scored by a model that never saw it). Held-out ROC-AUC = "
            f"<b>{d.auc:.3f}</b> (folds: {', '.join(f'{a:.3f}' for a in d.fold_auc)}; "
            f"Bonferroni-combined one-sided p = {_p(d.p_value)}). 0.5 means the samples are "
            "indistinguishable; values near 1 mean almost every row can be assigned to its "
            "sample.</p>"
        )
        if d.importances:
            parts.append(f'<img src="{data_uri(importance_chart(d.importances, d.auc))}" '
                         'alt="Domain-classifier permutation importances">')

    # ---- performance
    if report.performance is not None:
        pf = report.performance
        parts.append(
            "<h2>Model performance</h2><table><thead><tr><th>Metric</th>"
            '<th class="num">Reference</th><th class="num">Current</th>'
            '<th class="num">Ratio</th></tr></thead><tbody>'
            f'<tr><td>MAE</td><td class="num">{pf.mae_reference:.4f}</td>'
            f'<td class="num">{pf.mae_current:.4f}</td><td class="num">{pf.mae_ratio:.3f}</td></tr>'
            f'<tr><td>RMSE</td><td class="num">{pf.rmse_reference:.4f}</td>'
            f'<td class="num">{pf.rmse_current:.4f}</td>'
            f'<td class="num">{pf.rmse_ratio:.3f}</td></tr></tbody></table>'
            f'<p class="note">Test of H0: MAE<sub>current</sub> &le; {pf.max_ratio:.2f} &times; '
            f"MAE<sub>reference</sub>: p = {_p(pf.p_value)} "
            f"({'degraded' if pf.degraded else 'not degraded'} at alpha = {cfg.alpha:g}).</p>"
        )
        cur = report.current_data
        label = report.label.column if report.label is not None else None
        pred_cols = [c.column for c in report.columns if c.role == "prediction"]
        if cur is not None and label and pred_cols and report.time_column is not None:
            window = max(50, len(cur) // 10)
            if len(cur) >= 2 * window:
                roll = rolling_metrics(cur[label], cur[pred_cols[0]], window)
                img = data_uri(rolling_chart(roll, pf.mae_reference, pf.max_ratio))
                parts.append(f'<img src="{img}" alt="Rolling MAE over the current data">')

    # ---- method notes
    parts.append(
        "<h2>How to read this report</h2><ul class='note'>"
        "<li><b>Decision rule.</b> Every column gets one p-value from its decision test; these, "
        "the missing-rate tests and the domain classifier form one family, corrected with "
        f"<code>{escape(cfg.correction)}</code> at alpha = {cfg.alpha:g}.</li>"
        "<li><b>Drift is not damage.</b> A drifted input the model barely uses can be harmless, "
        "and a change in the input-output relationship (concept shift) is invisible to input "
        "tests. Check prediction drift and, once labels arrive, the performance section.</li>"
        "<li><b>On alarm:</b> look at the top rows and the domain-classifier importances, rule out "
        "a data-pipeline fault (constant values, unit changes, missing spikes) before retraining, "
        "and estimate the performance impact before acting.</li></ul>"
    )
    parts.append(
        f"<footer>Generated by driftwatch {escape(__version__)}. Config: bins = {cfg.n_bins}, "
        f"bootstrap = {cfg.n_bootstrap}, eps = {cfg.eps:g}, test = {escape(cfg.test)}, "
        f"psi_threshold = {cfg.psi_threshold}, min_psi = {cfg.min_psi:g}, "
        f"max_mae_ratio = {cfg.max_mae_ratio:g}, seed = {cfg.seed}.</footer>"
    )

    header = (
        '<header><p class="eyebrow">driftwatch report</p>'
        f"<h1>{escape(title)}</h1>"
        f'<p class="sub">Reference: <b>{escape(report.reference_name)}</b> '
        f"({report.n_reference:,} rows) &nbsp;&middot;&nbsp; Current: "
        f"<b>{escape(report.current_name)}</b> ({report.n_current:,} rows)</p></header>"
    )
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{header}{''.join(parts)}</main></body></html>\n"
    )
