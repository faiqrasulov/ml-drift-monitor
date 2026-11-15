import json
import subprocess
import sys

import pytest
from conftest import make_frame

from driftwatch.cli import EXIT_ALARM, EXIT_ERROR, EXIT_OK, main


@pytest.fixture()
def files(tmp_path):
    ref = tmp_path / "reference.parquet"
    make_frame(800, seed=1).to_parquet(ref, index=False)
    same = tmp_path / "same.csv"
    make_frame(300, seed=2).to_csv(same, index=False)
    shifted = tmp_path / "shifted.parquet"
    make_frame(300, seed=3, shift=1.0).to_parquet(shifted, index=False)
    return tmp_path, ref, same, shifted


def _args(ref, cur, *extra):
    return ["compare", str(ref), str(cur), "--target", "y", "--prediction", "pred",
            "--bootstrap", "200", *extra]


def test_exit_zero_without_drift(files, capsys):
    tmp, ref, same, _ = files
    assert main(_args(ref, same)) == EXIT_OK
    assert "NO ALARM" in capsys.readouterr().out


def test_exit_one_on_drift_and_writes_self_contained_html(files, capsys):
    tmp, ref, _, shifted = files
    html, js = tmp / "report.html", tmp / "summary.json"
    code = main(_args(ref, shifted, "--out", str(html), "--json", str(js), "--title", "T"))
    assert code == EXIT_ALARM
    out = capsys.readouterr().out
    assert "ALARM" in out and "x1" in out
    page = html.read_text()
    assert page.startswith("<!doctype html>") and "<style>" in page
    assert page.count("data:image/png;base64,") >= 2  # distributions + importances
    assert "<script" not in page and "<link" not in page and 'src="http' not in page
    summary = json.loads(js.read_text())
    assert summary["alarm"] is True and "x1" in summary["drifted_columns"]


def test_rule_of_thumb_mode_and_quiet(files, capsys):
    tmp, ref, _, shifted = files
    code = main(_args(ref, shifted, "--psi-threshold", "0.2", "--no-domain-classifier",
                      "--quiet"))
    assert code == EXIT_ALARM
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


def test_errors_exit_two(files, capsys):
    tmp, ref, same, _ = files
    assert main(_args(ref, tmp / "missing.parquet")) == EXIT_ERROR
    assert main(["compare", str(ref), str(same), "--target", "nope"]) == EXIT_ERROR
    bad = tmp / "data.txt"
    bad.write_text("x")
    assert main(["compare", str(ref), str(bad)]) == EXIT_ERROR
    assert "error" in capsys.readouterr().err


def test_console_entry_point_exit_code(files):
    tmp, ref, _, shifted = files
    proc = subprocess.run(
        [sys.executable, "-m", "driftwatch.cli", *_args(ref, shifted, "--quiet",
                                                         "--no-domain-classifier")],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == EXIT_ALARM, proc.stderr
