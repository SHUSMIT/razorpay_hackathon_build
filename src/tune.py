"""Optuna hyper-parameter search for all three model families.

The search strategy is deliberately asymmetric, and that asymmetry is the whole
point:

    SEARCH on a reduced train set: every fraud, plus 10% of the legitimate
    rows (~634k rows). Ranking one config against another does not need all 6M
    negatives, and dropping most of them turns a multi-hour study into minutes.
    What it DOES need is the positives -- a plain stratified 5% sample was
    measured here and it is actively misleading, leaving 508 frauds and ranking
    configs by noise.

    SCORE every trial on the FULL validation split. This is where cutting
    corners would actually cost us: PR-AUC on a rare-event problem is driven by
    the positive count, so scoring on a subsample would rank hyper-parameters
    by sampling noise. Val keeps all ~2,000 of its frauds.

    REFIT the winner on 100% of train (src/train.py), never on val.

The test split is not touched anywhere in this file.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from src.config import MODELS, RANDOM_SEED, REPORTS
from src.data import learn_categories, load_split, xy
from src.models import MODEL_NAMES, fit, predict, scale_pos_weight
from src.fmt import pct
from src.schema import FEATURES, LABEL

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SUBSAMPLE = 0.10   # of the NEGATIVES; every fraud is kept
DEFAULT_TRIALS = 30


def subsample(X, y, frac: float, seed: int = RANDOM_SEED, keep_all_positives: bool = True):
    """Shrink the search set without throwing away the signal.

    A plain stratified 5% sample was measured to be actively misleading here:
    it leaves only ~508 frauds, and a config tuned on 508 positives scored
    0.0056 where the same family on full train scored 0.0466. The ranking is
    driven by the positive count, so we keep EVERY fraud and subsample only the
    negatives. All 10,169 positives + 10% of negatives is ~634k rows -- fast to
    search, and faithful to how the config will rank at full scale.

    The base rate is deliberately inflated by this, so scale_pos_weight is
    recomputed from the sample inside each fit rather than carried over.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)
    pos = np.flatnonzero(y_arr == 1)
    neg = np.flatnonzero(y_arr == 0)
    if not keep_all_positives:
        pos = rng.choice(pos, size=max(int(round(len(pos) * frac)), 1), replace=False)
    neg = rng.choice(neg, size=max(int(round(len(neg) * frac)), 1), replace=False)
    sel = np.sort(np.concatenate([pos, neg]))
    return X.iloc[sel], y_arr[sel]


def suggest(trial: optuna.Trial, name: str) -> dict:
    if name == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 300),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
    if name == "catboost":
        return {
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "iterations": trial.suggest_int("iterations", 200, 800),
        }
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 500),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 20.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 150, 600),
    }


def tune_one(name: str, X_sub, y_sub, X_val, y_val, trials: int) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, name)
        model = fit(name, X_sub, y_sub, X_val, y_val, params=params)
        p = predict(name, model, X_val)
        return float(average_precision_score(y_val, p))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"{name}-pr-auc",
    )
    print(f"\n[tune] {name}: {trials} trials on {len(y_sub):,} subsampled rows "
          f"({int(np.sum(y_sub))} frauds), scored on {len(y_val):,} full val rows")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    print(f"[tune] {name}: best val PR-AUC {pct(study.best_value)}")
    print(f"[tune] {name}: best params {study.best_params}")
    out = {
        "model": name,
        "best_params": study.best_params,
        "best_val_pr_auc": float(study.best_value),
        "n_trials": trials,
        "subsample_frac": SUBSAMPLE,
        "subsample_rows": int(len(y_sub)),
        "subsample_frauds": int(np.sum(y_sub)),
        "val_rows": int(len(y_val)),
        "val_frauds": int(np.sum(y_val)),
        "trial_history": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ],
    }
    (MODELS / f"best_params_{name}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--subsample", type=float, default=SUBSAMPLE)
    ap.add_argument("--models", nargs="*", default=list(MODEL_NAMES))
    args = ap.parse_args()

    train = load_split("train", columns=FEATURES + [LABEL])
    cats = learn_categories(train)
    X_tr, y_tr = train[FEATURES], train[LABEL].to_numpy()
    del train
    X_val, y_val = xy("val", cats)
    y_val = y_val.to_numpy()

    X_sub, y_sub = subsample(X_tr, y_tr, args.subsample)
    print(f"[tune] train {len(y_tr):,} rows ({int(y_tr.sum())} frauds) "
          f"-> search sample {len(y_sub):,} rows ({int(y_sub.sum())} frauds)")
    del X_tr

    results = {}
    for name in args.models:
        results[name] = tune_one(name, X_sub, y_sub, X_val, y_val, args.trials)

    summary = {n: {"best_val_pr_auc": r["best_val_pr_auc"],
                   "best_params": r["best_params"]} for n, r in results.items()}
    (REPORTS / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[tune] summary (validation PR-AUC, search on "
          f"{args.subsample:.0%} of train)")
    for n, r in sorted(summary.items(), key=lambda kv: -kv[1]["best_val_pr_auc"]):
        print(f"    {n:<10} {pct(r['best_val_pr_auc'])}")
    print("[tune] wrote reports/tuning_summary.json")


if __name__ == "__main__":
    main()
