"""Split loading with consistent categorical encodings.

Parquet stores a `category` column's levels per file, so val and test can come
back with different category codes than train. A tree trained on train's codes
would then read val's categories as different values entirely -- a silent,
score-destroying bug. Every split is therefore re-levelled onto the categories
learned from train, and unseen levels collapse to NaN, which all three model
families handle natively.
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import DATA_PROCESSED, MODELS
from src.schema import CATEGORICAL, FEATURES, LABEL

CATEGORIES_FILE = MODELS / "categories.json"


def learn_categories(train: pd.DataFrame) -> dict[str, list[str]]:
    cats = {c: [str(v) for v in train[c].cat.categories] for c in CATEGORICAL}
    CATEGORIES_FILE.write_text(json.dumps(cats, indent=2), encoding="utf-8")
    return cats


def load_categories() -> dict[str, list[str]]:
    if CATEGORIES_FILE.exists():
        return json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))
    return {}


def align_categories(X: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    for c, levels in cats.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c].astype("string"), categories=levels)
    return X


def load_split(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = DATA_PROCESSED / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python run.py data` first")
    return pd.read_parquet(path, columns=columns)


def xy(name: str, cats: dict[str, list[str]] | None = None):
    """Feature matrix and label for one split, with aligned categoricals."""
    df = load_split(name, columns=FEATURES + [LABEL])
    y = df[LABEL].astype("int8")
    X = df[FEATURES].copy()
    if cats is None:
        cats = load_categories()
    if cats:
        X = align_categories(X, cats)
    return X, y


def catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants categorical columns as strings with no NaN."""
    out = X.copy()
    for c in CATEGORICAL:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna("__missing__")
    return out


def categorical_mask(X: pd.DataFrame) -> list[bool]:
    return [c in CATEGORICAL for c in X.columns]


def categorical_indices(X: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(X.columns) if c in CATEGORICAL]
