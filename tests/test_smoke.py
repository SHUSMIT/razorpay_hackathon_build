"""The end-to-end smoke test, run as part of the suite.

`scripts/smoke_test.py` is the thing you run before a demo; this makes CI run it
too, so a wiring break between training and serving fails the build rather than
the presentation. It is skipped rather than failed when the data splits are
absent, because a fresh clone has not downloaded them yet and a red suite on a
first checkout teaches people to ignore red suites.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not all((ROOT / "data" / "processed" / f"{s}.parquet").exists()
            for s in ("train", "val", "test")),
    reason="processed splits absent -- run `python run.py data`",
)


def test_end_to_end_smoke_test_is_green():
    """Runs the real script in a subprocess: exit code is the failure count."""
    proc = subprocess.run(
        [sys.executable, "scripts/smoke_test.py", "--rows", "10000"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{proc.returncode} smoke stage(s) failed\n\n"
            f"--- stdout ---\n{proc.stdout[-6000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}"
        )
    assert "SMOKE TEST GREEN" in proc.stdout


def test_smoke_test_writes_its_report():
    report = ROOT / "reports" / "smoke_test.json"
    assert report.exists(), "smoke test did not write reports/smoke_test.json"

    import json

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["failed"] == 0
    assert data["passed"] >= 15, "stages went missing from the smoke test"
    assert all(s["status"] in ("PASS", "SKIP") for s in data["stages"])
