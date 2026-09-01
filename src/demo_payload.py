"""Turn a stored demo row back into the transaction a gateway would have sent.

This lives outside the Streamlit app on purpose. The dashboard is the one part
of the system a person watches live, and the step most likely to break it is
this one -- it reverses the feature engineering, so it silently depends on
which columns `prepare.py` happens to write. Keeping it importable means the
test suite can catch that before a demo does.
"""
from __future__ import annotations

import pandas as pd


def clean(v):
    """Normalise pandas' many flavours of missing into None."""
    if v is None:
        return None
    if isinstance(v, float) and v != v:      # NaN
        return None
    if str(v) in ("nan", "NaT", "<NA>", ""):
        return None
    return str(v)


def to_payload(row: pd.Series | dict) -> dict:
    """Rebuild a scoring request from a demo row.

    The credit limit is recovered by inverting `amount_to_credit_limit`, since
    the raw limit is not stored -- it was dropped as an identity feature.

    merchant_state is deliberately absent: it was a degenerate shortcut and is
    no longer a feature (docs/FINDINGS.md), so it is not in the demo data
    either. Including it would render a permanent "unknown" on screen.
    """
    row = row if isinstance(row, dict) else row.to_dict()
    amount = float(row["abs_amount"])
    ratio = row.get("amount_to_credit_limit")
    credit_limit = (float(amount / ratio)
                    if ratio is not None and ratio == ratio and ratio > 0
                    else None)
    err = clean(row.get("error_kind"))
    has_chip = row.get("has_chip")
    return {
        "amount": amount,
        "date": str(row["date"]),
        "use_chip": clean(row.get("use_chip")),
        "mcc_category": clean(row.get("mcc_category")),
        "card_brand": clean(row.get("card_brand")),
        "card_type": clean(row.get("card_type")),
        "has_chip": bool(has_chip) if has_chip == has_chip and has_chip is not None
                    else None,
        "is_online": bool(row.get("is_online", 0)),
        "errors": None if err in (None, "none") else err,
        "credit_limit": credit_limit,
    }


def display_fields(payload: dict) -> dict:
    """What the operator sees about the transaction, in reading order."""
    return {
        "Amount": f"{payload['amount']:,.2f}",
        "When": str(payload["date"])[:16],
        "Presented": payload["use_chip"] or "unknown",
        "Card present": "no - online" if payload["is_online"] else "yes",
        "Merchant category": payload["mcc_category"] or "unknown",
        "Card": " ".join(x for x in [payload["card_brand"],
                                     payload["card_type"]] if x) or "unknown",
        "Terminal error": payload["errors"] or "none",
    }
