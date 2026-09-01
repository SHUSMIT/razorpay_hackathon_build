"""Explanations must be non-empty, stable, and mathematically sound."""
from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

from src.config import BASELINE_MODEL, TUNED_MODEL
from src.explain import explain_one, friendly, shap_contribs, top_factors, verify_additivity
from src.train import xy


@pytest.fixture(scope="module")
def booster():
    path = TUNED_MODEL if TUNED_MODEL.exists() else BASELINE_MODEL
    if not path.exists():
        pytest.skip("no model artefact -- run `make train`")
    b = xgb.Booster()
    b.load_model(path)
    return b


@pytest.fixture(scope="module")
def sample():
    X, y = xy("test")
    frauds = X[y == 1].head(20)
    return X.head(50), frauds


def test_explanation_is_non_empty(booster, sample):
    X, _ = sample
    factors = explain_one(booster, X.head(1))
    assert len(factors) == 3
    for f in factors:
        assert f["feature"] and f["label"]
        assert f["direction"] in {"increases risk", "decreases risk"}
        assert isinstance(f["shap_value"], float)


def test_top_features_are_stable_across_runs(booster, sample):
    """Same input twice must produce the same top-3, in the same order."""
    X, _ = sample
    for i in range(5):
        row = X.iloc[[i]]
        a = [f["feature"] for f in explain_one(booster, row)]
        b = [f["feature"] for f in explain_one(booster, row)]
        assert a == b, f"row {i}: {a} != {b}"


def test_explanations_differ_between_different_transactions(booster, sample):
    """A model that returns the same three features for everything is not
    explaining anything."""
    X, frauds = sample
    tops = {tuple(f["feature"] for f in explain_one(booster, X.iloc[[i]])) for i in range(20)}
    assert len(tops) > 1, "identical explanation for every transaction"


def test_shap_additivity(booster, sample):
    """contributions + bias == raw margin. Catches any wiring mistake."""
    X, _ = sample
    assert verify_additivity(booster, X) < 1e-3


def test_fraud_rows_have_risk_increasing_drivers(booster):
    """Known fraud must attract far more risk-increasing evidence than
    legitimate traffic does.

    Note the sign is NOT expected to be positive in absolute terms: the model
    carries a large positive bias (scale_pos_weight ~599 pushes the base
    log-odds well above zero), so a fraud row can score high while its net
    contribution sum is still negative. What must hold is the separation --
    and that the raw margin, bias included, is positive for most fraud.
    """
    X, y = xy("test")
    frauds, legit = X[y == 1].head(60), X[y == 0].head(600)

    f_contribs, bias = shap_contribs(booster, frauds)
    l_contribs, _ = shap_contribs(booster, legit)
    f_sum, l_sum = f_contribs.sum(axis=1), l_contribs.sum(axis=1)

    assert f_sum.mean() > l_sum.mean() + 5, \
        f"fraud {f_sum.mean():.2f} vs legit {l_sum.mean():.2f} -- no separation"
    # Margin = contributions + bias; positive margin means P(fraud) > 0.5.
    assert ((f_sum + bias) > 0).mean() > 0.7, "most known fraud should score above 0.5"
    assert ((l_sum + bias) < 0).mean() > 0.95, "legitimate traffic should score below 0.5"


def test_top_factors_ordered_by_absolute_impact(booster, sample):
    X, _ = sample
    contribs, _ = shap_contribs(booster, X.head(1))
    factors = top_factors(contribs[0], list(X.columns), k=5)
    magnitudes = [abs(f["shap_value"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert magnitudes[0] == pytest.approx(np.abs(contribs[0]).max(), abs=1e-5)


def test_labels_do_not_invent_meaning_for_anonymised_features():
    """V1..V28 are anonymised PCA components; claiming they mean something
    concrete would be a lie on camera."""
    assert friendly("V7") == "anonymised behaviour component V7"
    assert friendly("Amount") == "transaction amount"
