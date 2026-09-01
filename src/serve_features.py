"""Turn one raw transaction into the model's feature row.

The API accepts a transaction the way a payment gateway would actually send it
-- an amount, a timestamp, a merchant category code, whatever card and terminal
context happens to be available -- and derives the model's features here, using
the same transforms and the same train-period aggregates as src/prepare.py.

Real traffic is messy and incomplete, so every field except the amount is
optional and every derivation states what it does when the input is missing.
Nothing is silently invented: a missing value that the model can read as
"unknown" is passed through as NaN, which all three model families handle
natively, and the API reports which fields it had to fall back on.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MODELS
from src.data import load_categories
from src.schema import CATEGORICAL, FEATURES, NUMERIC

MCC_CODES = MODELS / "mcc_codes.json"
AGG_STATS = DATA_PROCESSED / "agg_stats.json"


def _load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class ServingFeaturizer:
    def __init__(self):
        self.mcc = _load_json(MCC_CODES, {})
        agg = _load_json(AGG_STATS, {})
        self.mcc_median = agg.get("mcc_median", {})
        self.global_median = float(agg.get("global_median", 0.0) or 0.0)
        self.categories = load_categories()

    # ------------------------------------------------------------- helpers
    def category_for(self, mcc_code) -> str | None:
        if mcc_code is None:
            return None
        return self.mcc.get(str(int(mcc_code)))

    def known_levels(self, column: str) -> list[str]:
        return self.categories.get(column, [])

    # ----------------------------------------------------------- transform
    def build(self, raw: dict) -> tuple[pd.DataFrame, list[str]]:
        """Return (one-row feature frame, list of fields that were missing)."""
        missing: list[str] = []

        amount = float(raw.get("amount", 0.0) or 0.0)
        abs_amount = abs(amount)

        # Time of day. Absent timestamp -> NaN rather than a fabricated hour.
        hour = np.nan
        ts = raw.get("date")
        if ts:
            try:
                dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
                hour = float(dt.hour)
            except (TypeError, ValueError):
                missing.append("date")
        else:
            missing.append("date")

        # Card-not-present. The dataset encodes this as a missing ZIP, and the
        # caller may say so directly or leave both out.
        if raw.get("is_online") is not None:
            is_online = float(bool(raw["is_online"]))
        elif "zip" in raw:
            is_online = float(raw.get("zip") in (None, "", 0))
        else:
            use_chip = str(raw.get("use_chip") or "")
            is_online = 1.0 if "online" in use_chip.lower() else 0.0
            missing.append("zip")

        errors = raw.get("errors")
        has_error = float(bool(errors))
        error_kind = str(errors).split(",")[0].strip() if errors else "none"

        mcc_category = raw.get("mcc_category") or self.category_for(raw.get("mcc"))
        if not mcc_category:
            mcc_category = None
            missing.append("mcc")

        credit_limit = raw.get("credit_limit")
        if credit_limit in (None, "", 0):
            amount_to_credit_limit = np.nan
            missing.append("credit_limit")
        else:
            amount_to_credit_limit = abs_amount / float(credit_limit)

        med = self.mcc_median.get(mcc_category, self.global_median) or self.global_median
        amount_vs_mcc_median = abs_amount / med if med else np.nan

        merchant_state = raw.get("merchant_state")
        if not merchant_state:
            merchant_state = "ONLINE" if is_online else None
            if not is_online:
                missing.append("merchant_state")

        has_chip = raw.get("has_chip")
        row = {
            "abs_amount": abs_amount,
            "log_amount": float(np.log1p(abs_amount)),
            "amount_to_credit_limit": amount_to_credit_limit,
            "amount_vs_mcc_median": amount_vs_mcc_median,
            "hour": hour,
            "is_night": float(hour < 6) if not np.isnan(hour) else np.nan,
            "is_online": is_online,
            "has_error": has_error,
            "has_chip": float(bool(has_chip)) if has_chip is not None else np.nan,
            "use_chip": raw.get("use_chip"),
            "merchant_state": merchant_state,
            "mcc_category": mcc_category,
            "card_brand": raw.get("card_brand"),
            "card_type": raw.get("card_type"),
            "error_kind": error_kind,
        }
        for c in CATEGORICAL:
            if row[c] is None:
                missing.append(c)

        frame = pd.DataFrame([row], columns=FEATURES)
        for c in NUMERIC:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float32")
        for c in CATEGORICAL:
            levels = self.known_levels(c)
            frame[c] = pd.Categorical(frame[c].astype("string"),
                                      categories=levels or None)
        return frame, sorted(set(missing))
