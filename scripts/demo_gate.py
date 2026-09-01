#!/usr/bin/env python
"""Fire a burst of high-risk transactions at a running API and show the
block-rate gate holding the line.

This is the 10 seconds of the pitch video that proves the decisioning is
bounded rather than just a model behind HTTP.

    python run.py api          # terminal 1
    python scripts/demo_gate.py  # terminal 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, DATA_SAMPLE  # noqa: E402

RAW_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_frauds(n: int) -> pd.DataFrame:
    test = DATA_PROCESSED / "test.parquet"
    df = pd.read_parquet(test) if test.exists() else pd.read_csv(
        DATA_SAMPLE / "sample_transactions.csv"
    )
    frauds = df[df.Class == 1]
    if frauds.empty:
        raise SystemExit("no fraud rows available -- run `python run.py data`")
    return frauds.sample(n, replace=len(frauds) < n, random_state=7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    try:
        health = requests.get(f"{args.api}/health", timeout=5).json()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"API unreachable at {args.api} ({exc}) -- start `python run.py api`")

    print(f"model {health['model_version']}  block>={health['thresholds']['block']:.2f}  "
          f"gate cap {health['gate']['max_block_rate']:.0%} over "
          f"{health['gate']['max_window']} decisions")
    requests.post(f"{args.api}/gate/reset", timeout=5)
    print(f"\nsending {args.n} known-fraud transactions ...\n")
    print(f"{'#':>4} {'score':>8}  {'model wanted':<13} {'served':<8} gate")
    print("-" * 62)

    rows = load_frauds(args.n)
    served_blocks = downgrades = wanted_blocks = missed = 0
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        payload = {c: float(r[c]) for c in RAW_COLS if c in r}
        res = requests.post(f"{args.api}/score", json=payload, timeout=10).json()
        served_blocks += res["decision"] == "block"
        downgrades += res["gated"]
        wanted_blocks += res["raw_decision"] == "block"
        missed += res["raw_decision"] == "allow"
        flag = "<-- DOWNGRADED by block-rate gate" if res["gated"] else ""
        if i <= 12 or res["gated"] and downgrades <= 3 or i % 10 == 0:
            print(f"{i:>4} {res['risk_score']:>8.4f}  {res['raw_decision']:<13} "
                  f"{res['decision']:<8} {flag}")

    gate = requests.get(f"{args.api}/gate", timeout=5).json()
    print("-" * 62)
    print(f"\n  known-fraud requests sent : {args.n}")
    print(f"  model wanted to block     : {wanted_blocks}")
    print(f"  model scored as allow     : {missed}  "
          f"(genuine misses -- recall is ~76%, not 100%)")
    print(f"  actually auto-blocked     : {served_blocks}")
    print(f"  downgraded to review      : {downgrades}")
    print(f"  final rolling block rate  : {gate['block_rate']:.1%} "
          f"(cap {gate['max_block_rate']:.0%})")
    if downgrades:
        print("\n  The gate held. A model that suddenly wants to block everything "
              "\n  degrades into a human review queue, not a merchant outage.")
    else:
        print("\n  Gate did not fire -- increase --n or lower MAX_BLOCK_RATE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
