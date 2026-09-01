#!/usr/bin/env python
"""How optimistic is our random split?

Production scores FUTURE transactions. A random split lets the model train on
transactions that occurred after the ones it is tested on, which on this data
is a larger effect than every modelling choice in the project combined.

This script trains the same configuration two ways -- time-ordered and random,
with identical split sizes -- and reports the gap.

Run: python scripts/time_split_check.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS  # noqa: E402
from src.data_prep import download  # noqa: E402
from src.features import build_features  # noqa: E402
from src.train import BASELINE_PARAMS, scale_pos_weight  # noqa: E402

TEST_FRACTION = 0.15
ROUNDS = 160  # the shipped baseline's best_iteration, held fixed for fairness


def main() -> int:
    df = download().drop_duplicates().reset_index(drop=True)
    X, y = build_features(df), df["Class"].astype(int)
    n = len(df)
    cut = int((1 - TEST_FRACTION) * n)

    def fit_eval(tr_idx, te_idx, tag):
        spw = scale_pos_weight(y.iloc[tr_idx])
        d_tr = xgb.DMatrix(X.iloc[tr_idx], label=y.iloc[tr_idx],
                           feature_names=list(X.columns))
        d_te = xgb.DMatrix(X.iloc[te_idx], feature_names=list(X.columns))
        booster = xgb.train({**BASELINE_PARAMS, "scale_pos_weight": spw}, d_tr,
                            num_boost_round=ROUNDS)
        ap = float(average_precision_score(y.iloc[te_idx], booster.predict(d_te)))
        frauds = int(y.iloc[te_idx].sum())
        print(f"  {tag:<44} PR-AUC {ap:.4f}   (test frauds = {frauds})")
        return ap, frauds

    # Time-ordered: train on the earliest rows, test on the latest.
    order = df["Time"].sort_values().index.to_numpy()
    # Random: same sizes, so the only difference is the ordering principle.
    perm = np.random.default_rng(42).permutation(n)

    print(f"\nSame model config ({ROUNDS} rounds), same 85/15 sizes, "
          "only the split principle differs:\n")
    ap_time, n_time = fit_eval(order[:cut], order[cut:],
                               "TIME-ordered (train past -> test future)")
    ap_rand, n_rand = fit_eval(perm[:cut], perm[cut:], "RANDOM (our protocol)")

    gap = ap_rand - ap_time
    print(f"\n  Random split is {gap:+.4f} PR-AUC more optimistic.")
    print("  For scale: the entire 50-trial Optuna search moved test PR-AUC by")
    print(f"  -0.0090, and the 95% CI on a single PR-AUC here is about +/-0.087.")
    print(f"\n  Caveat: the time-ordered test window holds only {n_time} frauds,")
    print("  so this is directional, not decisive.")
    print("\n  We keep the random split for comparability with the published")
    print("  baselines for this benchmark. A production system must be validated")
    print("  time-ordered -- this is the first thing to change.")

    (REPORTS / "time_split_check.json").write_text(json.dumps({
        "time_ordered_pr_auc": ap_time, "random_pr_auc": ap_rand,
        "optimism": gap, "test_frauds_time": n_time, "test_frauds_random": n_rand,
        "rounds": ROUNDS, "test_fraction": TEST_FRACTION,
    }, indent=2), encoding="utf-8")
    print(f"\n[time-split] wrote {REPORTS / 'time_split_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
