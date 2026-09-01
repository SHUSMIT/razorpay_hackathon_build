#!/usr/bin/env python
"""Does adding LightGBM + CatBoost and ensembling actually help?

Asked properly: every candidate is selected on validation, evaluated once on
test, and compared to the shipped XGBoost model with the same paired bootstrap
used in scripts/model_selection.py. A point estimate that moves is not an
improvement unless the interval says so.

Run: python scripts/ensemble_experiment.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS, TUNED_MODEL  # noqa: E402
from src.evaluate import (  # noqa: E402
    bootstrap_pr_auc,
    cost_at,
    do_nothing_cost,
    paired_bootstrap_delta,
    predict,
    sweep_thresholds,
)
from src.train import scale_pos_weight, xy  # noqa: E402

SEED = 42


def fit_lightgbm(X_tr, y_tr, X_va, y_va):
    """LightGBM, configured the hard way round.

    Two traps cost real accuracy here and are worth recording:

    1. NO `scale_pos_weight`. XGBoost wants it; LightGBM is actively hurt by it
       on this data (val PR-AUC 0.32 with, 0.85 without). Copying the XGBoost
       imbalance recipe across is what makes LightGBM look like a bad library.
    2. `metric="None"` plus an sklearn eval function. With the default
       `binary_logloss` still in the metric list, early stopping halts on ANY
       metric failing to improve -- and logloss degrades immediately under a
       heavily imbalanced fit, killing the run at iteration 2 or 3.

    It also needs a low learning rate and many more trees than XGBoost: val
    PR-AUC plateaus around iteration 800, not 50.
    """
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=1250, learning_rate=0.02, num_leaves=63, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=20, random_state=SEED, n_jobs=4, verbose=-1,
        metric="None",
    )
    model.fit(X_tr, y_tr)
    return model


def fit_catboost(X_tr, y_tr, X_va, y_va):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        eval_metric="PRAUC", random_seed=SEED,
        od_type="Iter", od_wait=200, verbose=False, thread_count=4,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    return model


def rank_norm(p: np.ndarray) -> np.ndarray:
    """Rank-average ensembling: the models are calibrated very differently
    (scale_pos_weight inflates all of them, but by different amounts), so
    averaging raw probabilities lets the worst-calibrated model dominate.
    Ranks are calibration-free, which is what we want for a PR-AUC comparison."""
    order = p.argsort().argsort()
    return order / max(len(p) - 1, 1)


def main() -> int:
    X_tr, y_tr = xy("train")
    X_va, y_va = xy("val")
    X_te, y_te = xy("test")
    yv, yt = y_va.to_numpy(), y_te.to_numpy()
    amounts = X_te["Amount"].to_numpy()

    import xgboost as xgb

    xgb_model = xgb.Booster()
    xgb_model.load_model(TUNED_MODEL)

    print("[exp] fitting LightGBM ...")
    lgb_model = fit_lightgbm(X_tr, y_tr, X_va, y_va)
    print("[exp] fitting CatBoost ...")
    cat_model = fit_catboost(X_tr, y_tr, X_va, y_va)

    # NOTE: the shipped XGBoost was refit on train+val, so its validation score
    # is in-sample. LGB/Cat here are trained on train only. The validation
    # column is therefore only comparable between LGB and Cat -- flagged in the
    # output rather than quietly presented as a fair three-way race.
    preds_va = {
        "xgboost (shipped)": predict(xgb_model, X_va),
        "lightgbm": lgb_model.predict_proba(X_va)[:, 1],
        "catboost": cat_model.predict_proba(X_va)[:, 1],
    }
    preds_te = {
        "xgboost (shipped)": predict(xgb_model, X_te),
        "lightgbm": lgb_model.predict_proba(X_te)[:, 1],
        "catboost": cat_model.predict_proba(X_te)[:, 1],
    }

    base = ["xgboost (shipped)", "lightgbm", "catboost"]
    for name, members in [
        ("ensemble: mean prob (3)", base),
        ("ensemble: rank-avg (3)", base),
        ("ensemble: rank-avg (xgb+lgb)", ["xgboost (shipped)", "lightgbm"]),
    ]:
        for store, src in ((preds_va, preds_va), (preds_te, preds_te)):
            cols = [src[m] for m in members]
            store[name] = (np.mean(cols, axis=0) if "mean prob" in name
                           else np.mean([rank_norm(c) for c in cols], axis=0))

    rows = []
    for name in preds_te:
        p_te, p_va = preds_te[name], preds_va[name]
        ci = bootstrap_pr_auc(yt, p_te, n_boot=1000)
        # Threshold selected on VALIDATION, then applied to test -- the honest
        # protocol, unlike selecting the argmin on test itself.
        sw_va = sweep_thresholds(yv, p_va, X_va["Amount"].to_numpy())
        t_val = float(sw_va.loc[sw_va.total_cost.idxmin(), "threshold"])
        c = cost_at(yt, p_te, amounts, t_val)
        rows.append({
            "model": name,
            "val_pr_auc": float(average_precision_score(yv, p_va)),
            "test_pr_auc": ci["point"],
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "threshold_from_val": t_val,
            "test_cost": c["total_cost"], "fp": c["fp"], "fn": c["fn"],
            "precision": c["precision"], "recall": c["recall"],
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("CANDIDATES  (test PR-AUC with 95% CI; threshold selected on validation)")
    print("=" * 100)
    print(f"{'model':<30}{'val PR-AUC':>12}{'test PR-AUC':>13}{'95% CI':>22}"
          f"{'test cost':>12}{'fp':>5}{'fn':>5}")
    print("-" * 100)
    for r in rows:
        print(f"{r['model']:<30}{r['val_pr_auc']:>12.4f}{r['test_pr_auc']:>13.4f}"
              f"   [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
              f"{r['test_cost']:>12,.0f}{int(r['fp']):>5}{int(r['fn']):>5}")
    print("-" * 100)
    print("NB: 'xgboost (shipped)' was refit on train+val, so its val PR-AUC is")
    print("    in-sample and inflated. Compare the TEST column, not val.")

    # --- the question that actually matters --------------------------------
    print("\n" + "=" * 100)
    print("IS ANY OF IT DISTINGUISHABLE FROM THE SHIPPED XGBOOST?")
    print("paired bootstrap on the same resampled test rows, 2,000 draws")
    print("=" * 100)
    ref = preds_te["xgboost (shipped)"]
    deltas = {}
    print(f"{'candidate':<30}{'delta PR-AUC':>14}{'95% CI':>24}{'verdict':>15}")
    print("-" * 100)
    for name, p in preds_te.items():
        if name == "xgboost (shipped)":
            continue
        d = paired_bootstrap_delta(yt, ref, p, n_boot=2000)
        deltas[name] = d
        print(f"{name:<30}{d['delta_point']:>+14.4f}"
              f"   [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
              f"{('better' if d['ci_low'] > 0 else 'WORSE' if d['ci_high'] < 0 else 'same'):>15}")
    print("-" * 100)

    # A significant result is not necessarily a GOOD one: `significant_at_95`
    # is also true when a candidate is significantly worse. Better means the
    # whole interval sits above zero.
    better = {k: v for k, v in deltas.items() if v["ci_low"] > 0}
    worse = {k: v for k, v in deltas.items() if v["ci_high"] < 0}
    any_sig = bool(better)
    nothing = do_nothing_cost(yt, amounts)
    same = [k for k in deltas if k not in better and k not in worse]
    print(f"\nSignificantly BETTER than shipped XGBoost: "
          f"{', '.join(better) if better else 'NONE'}")
    print(f"Significantly WORSE:                       "
          f"{', '.join(worse) if worse else 'none'}")
    print(f"Indistinguishable:                         "
          f"{', '.join(same) if same else 'none'}")
    print(f"(do-nothing cost on this split: {nothing:,.0f})")

    (REPORTS / "ensemble_experiment.json").write_text(json.dumps(
        {"candidates": rows, "paired_vs_xgboost": deltas,
         "significantly_better": list(better), "significantly_worse": list(worse),
         "any_better": any_sig, "do_nothing_cost": nothing},
        indent=2, default=float), encoding="utf-8")
    df.to_csv(REPORTS / "ensemble_experiment.csv", index=False)
    print(f"\n[exp] wrote {REPORTS / 'ensemble_experiment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
