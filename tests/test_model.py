"""Model artefacts load, score deterministically, and beat a trivial baseline."""
from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.config import BASELINE_MODEL, MODELS, TUNED_MODEL
from src.evaluate import cost_at, do_nothing_cost, predict, sweep_thresholds
from src.features import build_features, feature_names
from src.train import scale_pos_weight, xy


@pytest.fixture(scope="module")
def model_and_data():
    path = TUNED_MODEL if TUNED_MODEL.exists() else BASELINE_MODEL
    if not path.exists():
        pytest.skip("no model artefact -- run `make train`")
    booster = xgb.Booster()
    booster.load_model(path)
    X, y = xy("test")
    return booster, X, y


def test_artifact_loads_with_expected_features(model_and_data):
    booster, X, _ = model_and_data
    assert list(X.columns) == feature_names()
    assert booster.num_features() == X.shape[1]


def test_prediction_is_deterministic(model_and_data):
    booster, X, _ = model_and_data
    sample = X.head(500)
    a, b = predict(booster, sample), predict(booster, sample)
    assert np.array_equal(a, b), "same input gave different scores"
    assert ((a >= 0) & (a <= 1)).all()


def test_reload_gives_identical_scores(model_and_data):
    """A round-trip through disk must not change a single score."""
    booster, X, _ = model_and_data
    path = MODELS / "_roundtrip_test.json"
    booster.save_model(path)
    reloaded = xgb.Booster()
    reloaded.load_model(path)
    try:
        assert np.allclose(predict(booster, X.head(300)), predict(reloaded, X.head(300)))
    finally:
        path.unlink(missing_ok=True)


def test_beats_random_by_a_wide_margin(model_and_data):
    booster, X, y = model_and_data
    ap = average_precision_score(y, predict(booster, X))
    base = float(y.mean())
    assert ap > 0.6, f"PR-AUC {ap:.3f} is too low to be a working model"
    assert ap < 0.999, f"PR-AUC {ap:.4f} is implausibly perfect -- suspect leakage"
    assert ap > 100 * base


def test_scale_pos_weight_matches_imbalance():
    _, y = xy("train")
    spw = scale_pos_weight(y)
    assert spw == pytest.approx((len(y) - y.sum()) / y.sum())
    assert spw > 100


def test_features_are_deterministic_and_finite():
    X, _ = xy("test")
    again = build_features(__import__("src.train", fromlist=["load_split"]).load_split("test"))
    assert X.equals(again)
    assert np.isfinite(X.to_numpy()).all()


def test_hour_features_are_in_range():
    X, _ = xy("test")
    assert X.hour_of_day.between(0, 24).all()
    assert X.hour_sin.between(-1, 1).all() and X.hour_cos.between(-1, 1).all()


# --------------------------------------------------------------- cost model
def test_cost_model_is_monotone_at_the_extremes(model_and_data):
    booster, X, y = model_and_data
    p, amounts, y_np = predict(booster, X), X.Amount.to_numpy(), y.to_numpy()
    # Threshold ~1.0 blocks nothing, so cost collapses to the do-nothing cost.
    assert cost_at(y_np, p, amounts, 1.01)["total_cost"] == pytest.approx(
        do_nothing_cost(y_np, amounts)
    )
    # Threshold 0 blocks everything: zero missed fraud, maximum friction.
    at_zero = cost_at(y_np, p, amounts, 0.0)
    assert at_zero["fn"] == 0 and at_zero["fn_cost"] == 0
    assert at_zero["block_rate"] == pytest.approx(1.0)


def test_cost_optimal_beats_naive_half(model_and_data):
    booster, X, y = model_and_data
    p, amounts, y_np = predict(booster, X), X.Amount.to_numpy(), y.to_numpy()
    sweep = sweep_thresholds(y_np, p, amounts)
    best = sweep.loc[sweep.total_cost.idxmin()]
    assert best.total_cost <= cost_at(y_np, p, amounts, 0.5)["total_cost"]
    assert best.total_cost < do_nothing_cost(y_np, amounts)
    assert 0.0 < best.threshold < 1.0
