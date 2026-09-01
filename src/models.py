"""The three model families and their ensemble.

XGBoost, CatBoost and sklearn's HistGradientBoosting. All three consume pandas
`category` dtype natively, so the named categorical columns in this dataset
(merchant category, card brand, terminal error type, ...) go in as categories
rather than being one-hot exploded or given a meaningless ordinal encoding.

The ensemble is a weighted average of the three probability outputs, with the
weights chosen on VALIDATION. Whether that ensemble is actually better than the
best single model is then tested, not assumed -- a point estimate that moves is
not an improvement unless a paired bootstrap says so.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (
    CATBOOST_MAX_CTR_COMPLEXITY,
    CATBOOST_RAM_LIMIT,
    MODELS,
    N_THREADS,
    RANDOM_SEED,
)
from src.data import catboost_frame, categorical_indices, categorical_mask

XGB_MODEL = MODELS / "xgboost.json"
CAT_MODEL = MODELS / "catboost.cbm"
HGB_MODEL = MODELS / "histgb.joblib"
ENSEMBLE_CONFIG = MODELS / "ensemble.json"

MODEL_NAMES = ("xgboost", "catboost", "histgb")


def scale_pos_weight(y) -> float:
    pos = int(np.sum(y))
    return float((len(y) - pos) / max(pos, 1))


# ------------------------------------------------------------------ xgboost
def default_xgb_params(y) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight(y),
        "seed": RANDOM_SEED,
        "nthread": N_THREADS,
    }


def fit_xgboost(X, y, X_val, y_val, params: dict | None = None,
                rounds: int = 400, early_stopping: int = 40,
                sample_weight=None):
    import xgboost as xgb

    params = {**default_xgb_params(y), **(params or {})}
    if sample_weight is None:
        params.setdefault("scale_pos_weight", scale_pos_weight(y))
    else:
        # The weight vector already carries the class correction (and the
        # recency decay). Applying scale_pos_weight on top would count the
        # imbalance twice.
        params.pop("scale_pos_weight", None)
    dtr = xgb.DMatrix(X, label=y, weight=sample_weight, enable_categorical=True)
    dva = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    booster = xgb.train(params, dtr, num_boost_round=rounds,
                        evals=[(dva, "val")], early_stopping_rounds=early_stopping,
                        verbose_eval=False)
    return booster


def predict_xgboost(booster, X) -> np.ndarray:
    import xgboost as xgb

    d = xgb.DMatrix(X, enable_categorical=True)
    best = getattr(booster, "best_iteration", None)
    if best is not None:
        return booster.predict(d, iteration_range=(0, best + 1))
    return booster.predict(d)


# ----------------------------------------------------------------- catboost
def default_cat_params(y) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "random_seed": RANDOM_SEED,
        "scale_pos_weight": scale_pos_weight(y),
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": N_THREADS,
        # --- memory guards ---
        # CatBoost is the only member that will consume whatever it is given.
        # On a 7.6GB laptop with ~1.5GB free that is fatal, so it gets a hard
        # ceiling and its categorical-combination depth is capped. Both cost a
        # little accuracy and buy a run that finishes.
        "used_ram_limit": CATBOOST_RAM_LIMIT,
        "max_ctr_complexity": CATBOOST_MAX_CTR_COMPLEXITY,
        "border_count": 128,
    }


def fit_catboost(X, y, X_val, y_val, params: dict | None = None,
                 early_stopping: int = 40, sample_weight=None):
    from catboost import CatBoostClassifier, Pool

    params = {**default_cat_params(y), **(params or {})}
    if sample_weight is None:
        params.setdefault("scale_pos_weight", scale_pos_weight(y))
    else:
        params.pop("scale_pos_weight", None)
    cat_idx = categorical_indices(X)
    tr = Pool(catboost_frame(X), y, weight=sample_weight, cat_features=cat_idx)
    va = Pool(catboost_frame(X_val), y_val, cat_features=cat_idx)
    model = CatBoostClassifier(**params)
    model.fit(tr, eval_set=va, early_stopping_rounds=early_stopping, verbose=False)
    return model


def predict_catboost(model, X) -> np.ndarray:
    return model.predict_proba(catboost_frame(X))[:, 1]


# ------------------------------------------------------------------- histgb
def default_hgb_params() -> dict:
    return {
        "max_iter": 400,
        "learning_rate": 0.1,
        "max_depth": None,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "n_iter_no_change": 30,
        "max_bins": 128,
        "validation_fraction": 0.1,
        "random_state": RANDOM_SEED,
    }


def fit_histgb(X, y, X_val=None, y_val=None, params: dict | None = None,
               sample_weight=None):
    from sklearn.ensemble import HistGradientBoostingClassifier

    params = {**default_hgb_params(), **(params or {})}
    model = HistGradientBoostingClassifier(
        categorical_features=categorical_mask(X), **params)
    # HistGB has no scale_pos_weight, so imbalance is always expressed as
    # sample weights. When the caller supplies a vector it already carries the
    # class correction plus recency decay.
    w = sample_weight if sample_weight is not None else         np.where(np.asarray(y) == 1, scale_pos_weight(y), 1.0)
    model.fit(X, y, sample_weight=w)
    return model


def predict_histgb(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- dispatch
FIT = {"xgboost": fit_xgboost, "catboost": fit_catboost, "histgb": fit_histgb}
PREDICT = {"xgboost": predict_xgboost, "catboost": predict_catboost,
           "histgb": predict_histgb}


def fit(name: str, X, y, X_val, y_val, params: dict | None = None,
        sample_weight=None):
    if name == "histgb":
        return fit_histgb(X, y, params=params, sample_weight=sample_weight)
    return FIT[name](X, y, X_val, y_val, params=params,
                     sample_weight=sample_weight)


def predict(name: str, model, X) -> np.ndarray:
    return PREDICT[name](model, X)


# -------------------------------------------------------------- persistence
def save(name: str, model) -> None:
    if name == "xgboost":
        model.save_model(str(XGB_MODEL))
    elif name == "catboost":
        model.save_model(str(CAT_MODEL))
    else:
        import joblib

        joblib.dump(model, HGB_MODEL)


def load(name: str):
    if name == "xgboost":
        import xgboost as xgb

        b = xgb.Booster()
        b.load_model(str(XGB_MODEL))
        return b
    if name == "catboost":
        from catboost import CatBoostClassifier

        m = CatBoostClassifier()
        m.load_model(str(CAT_MODEL))
        return m
    import joblib

    return joblib.load(HGB_MODEL)


def available() -> list[str]:
    paths = {"xgboost": XGB_MODEL, "catboost": CAT_MODEL, "histgb": HGB_MODEL}
    return [n for n, p in paths.items() if p.exists()]


# ----------------------------------------------------------------- ensemble
def fit_ensemble_weights(preds: dict[str, np.ndarray], y) -> dict[str, float]:
    """Pick blend weights on the VALIDATION split by a small grid search.

    Deliberately coarse: with three members, a fine search would be fitting
    noise on ~2,000 positives. Steps of 0.1 over the simplex is enough to find
    whether a blend helps at all, which is the only question being asked.
    """
    from sklearn.metrics import average_precision_score

    names = [n for n in MODEL_NAMES if n in preds]
    best, best_score = None, -1.0
    grid = np.arange(0, 11) / 10.0
    for w in _simplex(len(names), grid):
        blend = sum(wi * preds[n] for wi, n in zip(w, names))
        s = average_precision_score(y, blend)
        if s > best_score:
            best, best_score = w, s
    weights = {n: float(round(wi, 3)) for n, wi in zip(names, best)}
    print(f"[ensemble] val PR-AUC {best_score:.4f} with weights {weights}")
    return weights


def _simplex(k: int, grid: np.ndarray):
    if k == 1:
        yield (1.0,)
        return
    for w in grid:
        for rest in _simplex(k - 1, grid):
            if abs(w + sum(rest) - 1.0) < 1e-9:
                yield (w, *rest)
            elif k == 2:
                continue


def ensemble_predict(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.get(n, 0.0) for n in preds)
    if total <= 0:
        return np.mean(list(preds.values()), axis=0)
    return sum(weights.get(n, 0.0) * p for n, p in preds.items()) / total


def save_ensemble(weights: dict[str, float], meta: dict | None = None) -> None:
    ENSEMBLE_CONFIG.write_text(
        json.dumps({"weights": weights, **(meta or {})}, indent=2), encoding="utf-8")


def load_ensemble() -> dict:
    if ENSEMBLE_CONFIG.exists():
        return json.loads(ENSEMBLE_CONFIG.read_text(encoding="utf-8"))
    return {"weights": {}}
