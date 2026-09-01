"""Feature engineering. One function, used identically by training and by the
API, so a transaction is described the same way in both places."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES

RAW_V_COLS = [f"V{i}" for i in range(1, 29)]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map a raw transaction frame to the model's feature matrix.

    Deliberately shallow: V1..V28 are already PCA components, so there is
    nothing sensible to engineer on top of them. The only additions are on the
    two interpretable raw columns.
    """
    out = pd.DataFrame(index=df.index)
    for c in RAW_V_COLS:
        out[c] = df[c].astype("float64")

    amount = df["Amount"].astype("float64").clip(lower=0)
    out["Amount"] = amount
    # Amount is long-tailed (median ~22, max ~25691); the log form is what a
    # tree actually splits on cleanly and it keeps the API robust to outliers.
    out["log_amount"] = np.log1p(amount)

    # Time is seconds elapsed since the first transaction in the 2-day window.
    # Absolute elapsed time does not generalise, but time-of-day does -- fraud
    # concentrates in low-traffic night hours.
    seconds = df["Time"].astype("float64")
    hour = (seconds / 3600.0) % 24.0
    out["hour_of_day"] = hour
    # Cyclical encoding so 23:00 and 01:00 are close together.
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return out


def feature_names() -> list[str]:
    """Column order the model was trained on."""
    if FEATURE_NAMES.exists():
        return json.loads(FEATURE_NAMES.read_text(encoding="utf-8"))
    return RAW_V_COLS + ["Amount", "log_amount", "hour_of_day", "hour_sin", "hour_cos"]


def save_feature_names(names: list[str]) -> None:
    FEATURE_NAMES.write_text(json.dumps(names, indent=2), encoding="utf-8")
