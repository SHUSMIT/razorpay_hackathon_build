#!/usr/bin/env python
"""Cross-platform task runner. Same targets as the Makefile, but works on
Windows where `make` usually is not installed.

    python run.py data | tune | train | assess | context
    python run.py api | app | test | all
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

TASKS: dict[str, list[list[str]]] = {
    "setup":   [[PY, "-m", "pip", "install", "-r", "requirements.txt"]],
    "data":    [[PY, "-m", "src.prepare"]],
    "tune":    [[PY, "-m", "src.tune", "--trials", "30"]],
    "train":   [[PY, "-m", "src.train"]],
    "assess":  [[PY, "-m", "src.assess"]],
    "context": [[PY, "-m", "src.context"]],
    "test":    [[PY, "-m", "pytest", "-q", "tests"]],
    "api":     [[PY, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1",
                 "--port", "8000"]],
    "app":     [[PY, "-m", "streamlit", "run", "app/streamlit_app.py"]],
}
# The full pipeline, in dependency order. `context` runs after assess so the
# reviewer evidence is built against the model that actually ships.
TASKS["all"] = (TASKS["data"] + TASKS["tune"] + TASKS["train"]
                + TASKS["assess"] + TASKS["context"] + TASKS["test"])

HELP = """Fraud Risk Manager -- task runner

  setup     install pinned dependencies
  data      load the raw tables, join, clean, split by time, check leakage
  tune      Optuna search for all three model families
  train     fit all three on full train, blend them, calibrate
  assess    open the held-out test split once and report
  context   build the reviewer evidence (percentiles + category base rates)
  all       data -> tune -> train -> assess -> context -> test

  api       start the risk service on 127.0.0.1:8000
  app       start the dashboard (needs the api running)
  test      run the test suite

Typical first run:
    python run.py setup
    python run.py all
    python run.py api          # terminal 1
    python run.py app          # terminal 2
"""


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        return 0
    target = sys.argv[1]
    if target not in TASKS:
        print(f"unknown target {target!r}\n")
        print(HELP)
        return 2
    extra = sys.argv[2:]
    for i, cmd in enumerate(TASKS[target]):
        full = cmd + (extra if i == len(TASKS[target]) - 1 else [])
        print(f"\n>>> {' '.join(full)}\n", flush=True)
        rc = subprocess.call(full, cwd=ROOT)
        if rc != 0:
            print(f"\n[run] step failed with exit code {rc}", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
