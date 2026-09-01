"""Serving-time featurisation, calibration and metrics.

The theme is the one the project is built around: the service must degrade
honestly on incomplete input rather than inventing values, and every number it
reports must mean what it says.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration import Calibration, expected_calibration_error
from src.metrics import cost_at, do_nothing_cost, sweep_thresholds
from src.schema import CATEGORICAL, FEATURES
from src.serve_features import ServingFeaturizer

FULL = {"amount": 142.50, "date": "2019-06-14T02:31:00",
        "use_chip": "Online Transaction", "mcc": 5812, "credit_limit": 12000.0,
        "card_brand": "Visa", "card_type": "Credit", "has_chip": True,
        "zip": None, "errors": None}


@pytest.fixture(scope="module")
def fz():
    return ServingFeaturizer()


# --------------------------------------------------------------- featurising
def test_full_input_produces_every_feature_and_reports_nothing_missing(fz):
    X, missing = fz.build(FULL)
    assert list(X.columns) == FEATURES
    assert len(X) == 1
    assert missing == []


def test_sparse_input_degrades_instead_of_inventing_values(fz):
    """Amount alone must still score, with the gaps declared, not guessed."""
    X, missing = fz.build({"amount": 89.99, "use_chip": "Online Transaction"})
    assert list(X.columns) == FEATURES
    for field in ("credit_limit", "date", "mcc"):
        assert field in missing, f"{field} was silently filled in"
    assert np.isnan(X["amount_to_credit_limit"].iloc[0]), \
        "a missing credit limit must stay unknown, not become a number"
    assert np.isnan(X["hour"].iloc[0]), "a missing timestamp must not invent an hour"


def test_amount_is_the_only_hard_requirement(fz):
    X, missing = fz.build({"amount": 10.0})
    assert X["abs_amount"].iloc[0] == pytest.approx(10.0)
    assert len(missing) > 0


def test_missing_zip_is_read_as_card_not_present(fz):
    online, _ = fz.build({"amount": 50.0, "zip": None})
    instore, _ = fz.build({"amount": 50.0, "zip": 94103.0})
    assert online["is_online"].iloc[0] == 1.0
    assert instore["is_online"].iloc[0] == 0.0


def test_refund_is_scored_on_its_magnitude(fz):
    X, _ = fz.build({"amount": -77.0})
    assert X["abs_amount"].iloc[0] == pytest.approx(77.0)


def test_mcc_code_resolves_to_a_human_readable_category(fz):
    X, _ = fz.build(FULL)
    assert str(X["mcc_category"].iloc[0]) == "Eating Places and Restaurants"


def test_unseen_category_does_not_crash_and_is_not_faked(fz):
    X, _ = fz.build({**FULL, "card_brand": "NotARealBrand"})
    assert X["card_brand"].isna().iloc[0], \
        "an unseen level must become unknown, not be mapped onto a known one"


def test_categorical_columns_keep_the_trained_levels(fz):
    X, _ = fz.build(FULL)
    for c in CATEGORICAL:
        levels = fz.known_levels(c)
        if levels:
            assert list(X[c].cat.categories) == levels


# --------------------------------------------------------------- calibration
def test_calibration_is_monotone_so_ranking_cannot_change():
    rng = np.random.default_rng(0)
    s = rng.random(4000)
    y = (rng.random(4000) < s * 0.3).astype(int)
    cal = Calibration.fit(s, y)
    grid = np.linspace(0, 1, 200)
    mapped = cal.predict(grid)
    assert np.all(np.diff(mapped) >= -1e-9), "calibration reordered the scores"


def test_calibration_improves_the_expected_calibration_error():
    rng = np.random.default_rng(1)
    n = 20000
    true_p = rng.random(n) * 0.2
    y = (rng.random(n) < true_p).astype(int)
    # Badly scaled scores: right order, wrong magnitude -- the exact failure a
    # heavy positive class weight produces.
    raw = np.clip(true_p * 4.0, 0, 1)
    cal = Calibration.fit(raw[:n // 2], y[:n // 2])
    before = expected_calibration_error(raw[n // 2:], y[n // 2:])
    after = expected_calibration_error(cal.predict(raw[n // 2:]), y[n // 2:])
    assert after < before


def test_unfitted_calibration_is_the_identity():
    cal = Calibration()
    s = np.array([0.1, 0.5, 0.9])
    assert np.allclose(cal.predict(s), s)


def test_calibration_survives_a_save_load_round_trip(tmp_path, monkeypatch):
    # Redirected to a temp file: a test must never overwrite the calibrator the
    # service is actually serving with.
    import src.calibration as calmod

    monkeypatch.setattr(calmod, "CALIBRATOR_FILE", tmp_path / "calibrator.json")
    rng = np.random.default_rng(2)
    s = rng.random(2000)
    y = (rng.random(2000) < s * 0.3).astype(int)
    cal = Calibration.fit(s, y)
    cal.save()
    again = Calibration.load()
    assert again.fitted
    assert np.allclose(cal.predict(s[:50]), again.predict(s[:50]))


# ------------------------------------------------------------- cost modelling
def test_cost_model_charges_the_actual_amount_for_a_missed_fraud():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.0, 0.0, 0.0, 0.0])       # blocks nothing
    amounts = np.array([100.0, 250.0, 10.0, 10.0])
    out = cost_at(y, p, amounts, threshold=0.5)
    assert out["fn"] == 2
    assert out["fn_cost"] == pytest.approx(350.0)
    assert out["fp_cost"] == pytest.approx(0.0)


def test_blocking_everything_costs_the_friction_penalty_per_good_customer():
    y = np.array([1, 0, 0])
    p = np.array([1.0, 1.0, 1.0])
    amounts = np.array([100.0, 10.0, 10.0])
    out = cost_at(y, p, amounts, threshold=0.5)
    assert out["fp"] == 2
    assert out["fn_cost"] == pytest.approx(0.0)
    assert out["fp_cost"] > 0


def test_do_nothing_cost_is_the_sum_of_fraud_amounts():
    y = np.array([1, 0, 1])
    amounts = np.array([70.0, 999.0, 30.0])
    assert do_nothing_cost(y, amounts) == pytest.approx(100.0)


def test_a_perfect_model_beats_doing_nothing():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    amounts = np.array([500.0, 20.0, 500.0, 20.0, 20.0])
    best = cost_at(y, p, amounts, threshold=0.5)
    assert best["total_cost"] < do_nothing_cost(y, amounts)


def test_threshold_sweep_is_ordered_and_complete():
    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.05).astype(int)
    p = rng.random(500)
    amounts = rng.random(500) * 100
    sweep = sweep_thresholds(y, p, amounts)
    assert sweep["threshold"].is_monotonic_increasing
    assert sweep["recall"].iloc[0] >= sweep["recall"].iloc[-1], \
        "recall must fall as the threshold rises"
