"""Sample weighting: class imbalance and recency, in one weight vector.

Two problems are solved by the same mechanism.

CLASS IMBALANCE. Fraud is roughly 1 in 600 here. Without correction a model
minimises loss by predicting "legitimate" forever.

CONCEPT DRIFT. This dataset's fraud mechanism is not stationary. Measured
across the time-ordered splits:

    train (2010-2018-02)  82.9% of fraud is card-not-present, at a 1.02% rate
    val   (2018-02-11)     card-not-present carries 1.9% of fraud, at 0.03%

Fraud from 2011 is therefore evidence about a world that no longer exists, and
weighting it equally with last month's fraud teaches the model the wrong
pattern. But discarding old rows outright throws away millions of legitimate
transactions that still describe normal behaviour perfectly well.

The compromise is exponential time decay: every row keeps its place in the
training set, but its influence halves every RECENCY_HALF_LIFE_DAYS. Old data
still shapes the model, recent data shapes it more, and the half-life is
selected on validation rather than asserted.

The two weights MULTIPLY. A fraud from last month outweighs a fraud from 2011,
and both outweigh any legitimate transaction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def class_weights(y: np.ndarray) -> np.ndarray:
    """Positive rows scaled by negatives/positives; negatives left at 1.0."""
    y = np.asarray(y)
    pos = int(y.sum())
    spw = float((len(y) - pos) / max(pos, 1))
    return np.where(y == 1, spw, 1.0)


def recency_weights(dates: pd.Series, half_life_days: float | None,
                    reference: pd.Timestamp | None = None) -> np.ndarray:
    """Exponential decay by age. `None` or <=0 disables it (all weights 1.0).

    The reference point is the NEWEST training date, so weights describe age
    relative to the end of the training window rather than to today.
    """
    if not half_life_days or half_life_days <= 0:
        return np.ones(len(dates), dtype="float64")
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    ref = reference if reference is not None else d.max()
    age_days = (ref - d).dt.total_seconds().to_numpy() / 86400.0
    age_days = np.clip(age_days, 0.0, None)
    return np.power(0.5, age_days / float(half_life_days))


def combined_weights(y: np.ndarray, dates: pd.Series | None,
                     half_life_days: float | None,
                     reference: pd.Timestamp | None = None) -> np.ndarray:
    """Class weight x recency weight, normalised to mean 1.0.

    Normalisation keeps the effective sample size stable as the half-life
    changes, so learning rates and regularisation stay comparable between
    candidate half-lives instead of silently shrinking with the weights.
    """
    w = class_weights(y)
    if dates is not None:
        w = w * recency_weights(dates, half_life_days, reference)
    mean = float(w.mean())
    return w / mean if mean > 0 else w


def effective_sample_size(w: np.ndarray) -> float:
    """Kish's effective sample size: how many equally-weighted rows this is worth.

    Useful as a sanity check -- an aggressive half-life can quietly reduce
    millions of rows to the influence of a few thousand.
    """
    w = np.asarray(w, dtype="float64")
    s = w.sum()
    return float(s * s / np.sum(w * w)) if s > 0 else 0.0
