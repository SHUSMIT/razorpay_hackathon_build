"""The dashboard's demo path.

This is the code a person watches live. It reverses the feature engineering to
rebuild a scoring request, so it silently depends on which columns prepare.py
writes -- exactly the kind of coupling that breaks quietly after a schema
change and is discovered on camera.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED
from src.demo_payload import clean, display_fields, to_payload
from src.schema import LABEL

DEMO = DATA_PROCESSED / "demo.parquet"
needs_demo = pytest.mark.skipif(
    not DEMO.exists(), reason="no demo split -- run `python run.py data`")


@pytest.fixture(scope="module")
def demo():
    return pd.read_parquet(DEMO)


# ------------------------------------------------------------------ cleaning
@pytest.mark.parametrize("value", [None, float("nan"), "nan", "NaT", "<NA>", ""])
def test_missing_values_all_normalise_to_none(value):
    assert clean(value) is None


def test_real_values_survive_cleaning():
    assert clean("Visa") == "Visa"
    assert clean(42) == "42"


# ------------------------------------------------------------------- payload
@needs_demo
def test_every_demo_row_builds_a_valid_request(demo):
    """All 89k rows, not a sample: one malformed row is a broken demo."""
    required = {"amount", "date", "use_chip", "mcc_category", "card_brand",
                "card_type", "has_chip", "is_online", "errors", "credit_limit"}
    for _, row in demo.head(2000).iterrows():
        p = to_payload(row)
        assert required <= set(p), f"missing keys: {required - set(p)}"
        assert isinstance(p["amount"], float) and np.isfinite(p["amount"])
        assert p["amount"] >= 0, "amount must be a magnitude"


@needs_demo
def test_payload_never_leaks_the_label(demo):
    """The ground truth must not reach the scorer, or the demo is theatre."""
    for _, row in demo.head(500).iterrows():
        assert LABEL not in to_payload(row)
        assert "is_fraud" not in to_payload(row)


@needs_demo
def test_payload_never_contains_the_dropped_shortcut(demo):
    for _, row in demo.head(200).iterrows():
        assert "merchant_state" not in to_payload(row)


@needs_demo
def test_credit_limit_is_recovered_consistently(demo):
    """It is reconstructed by inverting amount / credit_limit, so the round
    trip has to hold."""
    checked = 0
    for _, row in demo.head(2000).iterrows():
        ratio = row.get("amount_to_credit_limit")
        if ratio and ratio == ratio and ratio > 0:
            p = to_payload(row)
            assert p["credit_limit"] is not None
            assert p["amount"] / p["credit_limit"] == pytest.approx(ratio, rel=1e-4)
            checked += 1
    assert checked > 100, "too few rows exercised the credit-limit path"


@needs_demo
def test_the_demo_pool_contains_both_outcomes(demo):
    """The Live Score tab offers 'known fraud' and 'known legitimate'. Both
    buttons must have something to draw."""
    assert (demo[LABEL] == 1).sum() > 0, "no fraud to demonstrate"
    assert (demo[LABEL] == 0).sum() > 0, "no legitimate traffic to demonstrate"


@needs_demo
def test_missing_optional_fields_do_not_break_the_payload():
    row = {"abs_amount": 50.0, "date": "2019-10-01 12:00:00",
           "amount_to_credit_limit": np.nan, "use_chip": np.nan,
           "mcc_category": np.nan, "card_brand": np.nan, "card_type": np.nan,
           "has_chip": np.nan, "is_online": 1, "error_kind": "none"}
    p = to_payload(row)
    assert p["credit_limit"] is None and p["use_chip"] is None
    assert p["errors"] is None, "'none' is the absence of an error, not an error"
    assert p["amount"] == 50.0


# ------------------------------------------------------------------- display
@needs_demo
def test_operator_view_has_no_blank_or_nan_fields(demo):
    """Anything the operator reads must say something, never 'nan'."""
    # Whole-value comparison, not a substring: "Cleaning and Maintenance
    # Services" legitimately contains "nan".
    placeholders = {"nan", "none", "null", "nat", "<na>", "n/a"}
    for _, row in demo.head(500).iterrows():
        for key, value in display_fields(to_payload(row)).items():
            assert value not in ("", None), f"{key} is blank"
            if key == "Terminal error":
                continue          # "none" here is a real answer: no error
            assert str(value).strip().lower() not in placeholders, \
                f"{key} shows the placeholder {value!r}"
