"""Command line interface: ``driftwatch compare reference current [options]``.

Exit codes: 0 = no alarm, 1 = drift / label drift / performance alarm, 2 = usage or input error.
The non-zero exit on drift lets the command gate a CI job or a scheduled pipeline.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from driftwatch import __version__
from driftwatch.monitor import DriftConfig, DriftMonitor, DriftReport

EXIT_OK, EXIT_ALARM, EXIT_ERROR = 0, 1, 2


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a .parquet or .csv file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported file type {suffix!r} (use .parquet or .csv)")


def _split(arg: str | None) -> list[str] | None:
    return [s.strip() for s in arg.split(",") if s.strip()] if arg else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftwatch", description="Data, prediction and performance drift monitoring."
    )
    parser.add_argument("--version", action="version", version=f"driftwatch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "compare",
        help="compare a current data set against a reference",
        description="Compare CURRENT against REFERENCE. Exit code 0 = no alarm, 1 = alarm, "
        "2 = error.",
    )
    p.add_argument("reference", help="reference data (.parquet or .csv)")
    p.add_argument("current", help="current data (.parquet or .csv)")
    p.add_argument("--target", help="label column (enables label-drift and performance checks)")
    p.add_argument("--prediction", help="model prediction column (prediction drift)")
    p.add_argument("--features", help="comma-separated feature columns (default: all others)")
    p.add_argument("--categorical", help="comma-separated categorical columns (default: inferred)")
    p.add_argument("--exclude", help="comma-separated columns to ignore (e.g. ids)")
    p.add_argument("--time-column",
                   help="timestamp column: excluded from tests, orders the rolling-MAE chart")
    p.add_argument("--out", help="write the HTML report here")
    p.add_argument("--json", dest="json_path", help="write the JSON summary here")
    p.add_argument("--title", help="report title")
    p.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")
    p.add_argument("--correction", choices=["bh", "bonferroni", "none"], default="bh")
    p.add_argument("--test", choices=["psi", "ks"], default="psi",
                   help="psi = bootstrap-calibrated PSI (default); ks = KS / chi-square")
    p.add_argument("--psi-threshold", type=float, default=None,
                   help="use a fixed PSI cut-off (e.g. 0.2) instead of calibrated p-values")
    p.add_argument("--bins", type=int, default=10, help="PSI bins (reference quantiles)")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap replicates for PSI")
    p.add_argument("--min-psi", type=float, default=0.0,
                   help="practical-significance floor: ignore significant drift below this PSI")
    p.add_argument("--no-domain-classifier", action="store_true",
                   help="skip the multivariate domain-classifier test")
    p.add_argument("--max-mae-ratio", type=float, default=1.10,
                   help="performance alarm above this multiple of the reference MAE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="print only the verdict line")
    return parser


def summary_text(report: DriftReport, verbose: bool = True) -> str:
    """Plain-text summary for the terminal."""
    head = "ALARM" if report.alarm else "NO ALARM"
    lines = [
        f"driftwatch {__version__}: {report.current_name} ({report.n_current:,} rows) vs "
        f"{report.reference_name} ({report.n_reference:,} rows) -> {head}"
    ]
    if not verbose:
        return lines[0]
    lines += [f"  - {a}" for a in report.alarms()]
    if report.alarm:
        lines.append(f"Likely cause: {report.triage.headline}")
        lines += [f"  * {i}" for i in report.pipeline_issues]
    lines.append("")
    lines.append(f"{'column':<22}{'role':<11}{'PSI':>8}{'KS D':>8}{'p adj':>9}  status")
    rows = list(report.columns) + ([report.label] if report.label is not None else [])
    for c in rows:
        ks = f"{c.ks:.3f}" if c.ks is not None else "-"
        padj = "-" if c.p_adjusted is None else f"{c.p_adjusted:.3g}"
        status = "DRIFT" if c.drift else ("MISSING" if c.missing_drift else "ok")
        lines.append(f"{c.column[:21]:<22}{c.role:<11}{c.psi:>8.3f}{ks:>8}{padj:>9}  {status}")
    if report.domain is not None:
        lines.append(f"\ndomain classifier AUC = {report.domain.auc:.3f} "
                     f"(p adj = {report.domain_p_adjusted:.3g})")
    if report.performance is not None:
        pf = report.performance
        lines.append(f"MAE {pf.mae_current:.4f} vs reference {pf.mae_reference:.4f} "
                     f"({pf.mae_ratio:.2f}x, {'degraded' if pf.degraded else 'ok'})")
    return "\n".join(lines)


def run_compare(args: argparse.Namespace) -> int:
    reference = read_table(args.reference)
    current = read_table(args.current)
    exclude = set(_split(args.exclude) or [])
    reference = reference.drop(columns=[c for c in exclude if c in reference.columns])
    current = current.drop(columns=[c for c in exclude if c in current.columns])
    config = DriftConfig(
        alpha=args.alpha,
        correction=args.correction,
        test=args.test,
        psi_threshold=args.psi_threshold,
        n_bins=args.bins,
        n_bootstrap=args.bootstrap,
        min_psi=args.min_psi,
        domain_classifier=not args.no_domain_classifier,
        max_mae_ratio=args.max_mae_ratio,
        seed=args.seed,
    )
    monitor = DriftMonitor(config).fit(
        reference,
        features=_split(args.features),
        categorical=_split(args.categorical),
        target=args.target,
        prediction=args.prediction,
        time_column=args.time_column,
        name=Path(args.reference).name,
    )
    report = monitor.check(current, name=Path(args.current).name)
    if args.out:
        report.to_html(args.out, title=args.title)
    if args.json_path:
        report.to_json(args.json_path)
    print(summary_text(report, verbose=not args.quiet))
    if args.out and not args.quiet:
        print(f"HTML report: {args.out}")
    return EXIT_ALARM if report.alarm else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_compare(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"driftwatch: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
