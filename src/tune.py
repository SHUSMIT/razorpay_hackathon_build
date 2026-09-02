"""Optuna hyper-parameter search for all three model families.

The search strategy is deliberately asymmetric, and that asymmetry is the whole
point:

    SEARCH on a reduced train set: every fraud, plus 10% of the legitimate
    rows (~634k rows). Ranking one config against another does not need all 6M
    negatives, and dropping most of them turns a multi-hour study into minutes.
    What it DOES need is the positives -- a plain stratified 5% sample was
    measured here and it is actively misleading, leaving 508 frauds and ranking
    configs by noise.

    SCORE every trial on val_FIT -- the earlier 60% of validation. Not the
    full split: val_sel is what src/assess.py uses to choose the operating
    threshold, and selecting hyper-parameters against it would leave that
    threshold tuned to data its own model was fitted around. The cost is a
    noisier ranking signal (688 frauds rather than 1,243), and that is the
    right trade: the threshold slice stays genuinely untouched.

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

from src.config import MODELS, OPTUNA_DB, RANDOM_SEED, REPORTS, available_ram_gb
from src.data import learn_categories, load_split, xy
from src.models import MODEL_NAMES, fit, predict
from src.weights import combined_weights, effective_sample_size
from src.fmt import pct
from src.schema import FEATURES, LABEL

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SUBSAMPLE = 0.10   # of the NEGATIVES; every fraud is kept
DEFAULT_TRIALS = 30
# Stored inside best_params but consumed by the weighting code, not by the
# model, so it is popped back out before any params dict reaches a booster.
HALF_LIFE_KEY = "recency_half_life_days"


def subsample(X, y, frac: float, seed: int = RANDOM_SEED, keep_all_positives: bool = True,
              dates=None):
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
    if dates is not None:
        return X.iloc[sel], y_arr[sel], dates.iloc[sel]
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


def _val_fit_cut(n: int) -> int:
    """Row index where val_fit ends, matching src/train.py's split."""
    path = MODELS / "val_split.json"
    if path.exists():
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["cut_index"])
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return int(n * 0.60)


def _progress(name: str, total: int):
    """Print each trial as it lands, so a long study is not a silent black box."""

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        ram = available_ram_gb()
        ram_note = f"  ram_free={ram:.1f}gb" if ram is not None else ""
        best = study.best_value if study.best_trial else float("nan")
        print(f"    [{name}] trial {done}/{total}  this={pct(trial.value)}  "
              f"best={pct(best)}{ram_note}", flush=True)

    return cb


def tune_one(name: str, X_sub, y_sub, X_val, y_val, trials: int,
             dates_sub=None) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, name)
        # How fast old evidence should stop counting is not something to assert
        # -- this dataset's fraud mechanism drifts, so the decay rate is tuned
        # alongside everything else. 0 means no decay (all history equal).
        half_life = trial.suggest_categorical(
            HALF_LIFE_KEY, [0, 30, 60, 90, 120, 180, 365, 730])
        w = combined_weights(y_sub, dates_sub, half_life)
        try:
            model = fit(name, X_sub, y_sub, X_val, y_val, params=params,
                        sample_weight=w)
            p = predict(name, model, X_val)
        except Exception as exc:  # noqa: BLE001 - MemoryError included
            # One greedy configuration must not take the whole study with it.
            # Pruning the trial lets the sampler learn to avoid that corner.
            print(f"    [{name}] trial failed ({type(exc).__name__}: "
                  f"{str(exc)[:90]}) -- pruned", flush=True)
            raise optuna.TrialPruned() from exc
        return float(average_precision_score(y_val, p))

    # SQLite-backed so an interrupted search RESUMES instead of restarting from
    # trial zero. On a memory-tight laptop that is not a nicety: it is the
    # difference between losing twenty minutes and losing three hours.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"{name}-pr-auc",
        storage=f"sqlite:///{OPTUNA_DB.as_posix()}",
        load_if_exists=True,
    )
    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(trials - done, 0)
    print(f"\n[tune] {name}: {len(y_sub):,} search rows "
          f"({int(np.sum(y_sub))} frauds), scored on {len(y_val):,} val_fit rows")
    if done:
        print(f"[tune] {name}: resuming -- {done} completed trial(s) found in "
              f"{OPTUNA_DB.name}, {remaining} to go")
    if remaining == 0:
        print(f"[tune] {name}: study already complete, skipping")
    else:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False,
                       callbacks=[_progress(name, trials)], gc_after_trial=True)

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
    ap.add_argument("--force", action="store_true",
                    help="re-tune even if best_params already exist")
    args = ap.parse_args()

    train = load_split("train", columns=FEATURES + [LABEL, "date"])
    cats = learn_categories(train)
    X_tr, y_tr = train[FEATURES], train[LABEL].to_numpy()
    dates_tr = train["date"]
    del train
    X_val, y_val = xy("val", cats)
    y_val = y_val.to_numpy()

    # Trials are scored on val_FIT only, never on val_sel. val_sel is what
    # src/assess.py uses to choose the operating threshold, and if the
    # hyper-parameters had been selected using it, that threshold would be
    # tuned against data its own model had already been fitted around. Fewer
    # positives here (688 rather than 1,243) makes the ranking noisier, which
    # is the price of keeping the threshold slice genuinely untouched.
    cut = _val_fit_cut(len(y_val))
    X_val, y_val = X_val.iloc[:cut], y_val[:cut]
    print(f"[tune] scoring trials on val_fit: {len(y_val):,} rows, "
          f"{int(y_val.sum())} frauds (val_sel is held back for the threshold)")

    X_sub, y_sub, dates_sub = subsample(X_tr, y_tr, args.subsample,
                                        dates=dates_tr)
    print(f"[tune] train {len(y_tr):,} rows ({int(y_tr.sum())} frauds) "
          f"-> search sample {len(y_sub):,} rows ({int(y_sub.sum())} frauds)")
    print(f"[tune] recency decay is a tuned dimension; half-life candidates "
          f"(days): 0 (off), 30, 60, 90, 120, 180, 365, 730")
    del X_tr, dates_tr

    results = {}
    for name in args.models:
        done_file = MODELS / f"best_params_{name}.json"
        if done_file.exists() and not args.force:
            print(f"[tune] {name}: {done_file.name} already exists, skipping "
                  f"(use --force to redo)")
            results[name] = json.loads(done_file.read_text(encoding="utf-8"))
            continue
        results[name] = tune_one(name, X_sub, y_sub, X_val, y_val, args.trials,
                                 dates_sub=dates_sub)

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
