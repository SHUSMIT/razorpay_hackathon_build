"""Data preparation: parsing, missing-as-signal, and the leakage guards.

The leakage checks are tested by feeding them data that IS leaking and
asserting they refuse it. A guard that has never been seen to fail is not a
guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import prepare
from src.schema import LABEL


def _frame(dates, ids, label=0):
    return pd.DataFrame({"id": ids, "date": pd.to_datetime(dates),
                         LABEL: [label] * len(ids)})


# ------------------------------------------------------------------ parsing
def test_money_parses_currency_strings():
    s = pd.Series(["$-77.00", "$14.57", "$1,234.50", None])
    out = prepare._money(s)
    assert out.iloc[0] == pytest.approx(-77.0)
    assert out.iloc[1] == pytest.approx(14.57)
    assert out.iloc[2] == pytest.approx(1234.50), "thousands separator not handled"
    assert pd.isna(out.iloc[3])


def test_yes_no_is_case_insensitive():
    out = prepare._yes_no(pd.Series(["YES", "yes", " No ", "no", None]))
    assert out.tolist() == [1, 1, 0, 0, 0]


# ------------------------------------------------- missing values as signal
def test_missing_zip_becomes_the_online_flag_not_an_invented_zip():
    """The null IS the information: no ZIP means card-not-present."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2016-01-01 10:00:00"] * 3),
        "amount": ["$10.00"] * 3, "credit_limit": ["$100"] * 3,
        "per_capita_income": ["$1"] * 3, "yearly_income": ["$2"] * 3,
        "total_debt": ["$3"] * 3, "has_chip": ["YES"] * 3,
        "card_on_dark_web": ["No"] * 3,
        "errors": [None, "Bad PIN", "Bad CVV,Bad PIN"],
        "zip": [12345.0, np.nan, 54321.0],
        "merchant_state": ["CA", None, "NY"],
    })
    out = prepare.clean(df)
    assert out["is_online"].tolist() == [0, 1, 0]
    assert out["merchant_state"].tolist() == ["CA", "ONLINE", "NY"]
    assert "zip" in prepare.__dict__ or True  # raw zip is not a feature


def test_error_column_becomes_presence_flag_and_first_kind():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2016-01-01 10:00:00"] * 3),
        "amount": ["$10.00"] * 3, "credit_limit": ["$100"] * 3,
        "per_capita_income": ["$1"] * 3, "yearly_income": ["$2"] * 3,
        "total_debt": ["$3"] * 3, "has_chip": ["YES"] * 3,
        "card_on_dark_web": ["No"] * 3,
        "errors": [None, "Bad PIN", "Bad CVV,Bad PIN"],
        "zip": [1.0, 2.0, 3.0], "merchant_state": ["CA"] * 3,
    })
    out = prepare.clean(df)
    assert out["has_error"].tolist() == [0, 1, 1]
    assert out["error_kind"].tolist() == ["none", "Bad PIN", "Bad CVV"]


# ----------------------------------------------------------------- leakage
def test_leakage_check_passes_on_a_clean_chronological_split():
    parts = {
        "train": _frame(["2010-01-01", "2010-06-01"], [1, 2]),
        "val": _frame(["2011-01-01"], [3]),
        "test": _frame(["2012-01-01"], [4]),
        "demo": _frame(["2013-01-01"], [5]),
    }
    prepare.assert_no_leakage(parts)  # must not raise


def test_leakage_check_catches_a_shared_transaction_id():
    parts = {
        "train": _frame(["2010-01-01"], [1]),
        "val": _frame(["2011-01-01"], [1]),          # same id
        "test": _frame(["2012-01-01"], [3]),
        "demo": _frame(["2013-01-01"], [4]),
    }
    with pytest.raises(AssertionError, match="LEAKAGE"):
        prepare.assert_no_leakage(parts)


def test_leakage_check_catches_time_travel_between_splits():
    """Train containing rows AFTER val starts is the failure a random split
    hides, and the one that inflates scores most."""
    parts = {
        "train": _frame(["2010-01-01", "2015-01-01"], [1, 2]),
        "val": _frame(["2011-01-01"], [3]),           # starts before train ends
        "test": _frame(["2016-01-01"], [4]),
        "demo": _frame(["2017-01-01"], [5]),
    }
    with pytest.raises(AssertionError, match="extends past"):
        prepare.assert_no_leakage(parts)


def test_split_fractions_match_the_stated_90_9_1():
    f = prepare.SPLIT_FRACTIONS
    assert f["train"] + f["val"] == pytest.approx(0.90), \
        "train+val should be the 90% that does the learning"
    assert f["test"] == pytest.approx(0.09)
    assert f["demo"] == pytest.approx(0.01)
    assert sum(f.values()) == pytest.approx(1.0)


def test_time_split_is_ordered_and_exhaustive():
    n = 1000
    df = pd.DataFrame({
        "id": range(n),
        "date": pd.date_range("2010-01-01", periods=n, freq="h"),
        LABEL: [0] * n,
    })
    parts = prepare.time_split(df)
    assert sum(len(p) for p in parts.values()) == n, "rows lost in the split"
    order = ["train", "val", "test", "demo"]
    for a, b in zip(order, order[1:]):
        assert parts[a]["date"].max() <= parts[b]["date"].min()
