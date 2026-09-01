"""Dataset schema: what we feed the model, what we drop, and what a human calls it.

This file is the reason the pipeline is not welded to one dataset. Everything
downstream -- feature building, training, the API, the reviewer UI -- reads the
column roles from here rather than hard-coding a column list.

The predecessor pipeline was trained on anonymised PCA components (V1..V28)
whose meanings were never published. A reviewer could be shown how UNUSUAL a
value was, but never what it MEANT. This dataset has real named columns, so
the same explanation machinery now produces something a human can act on:

    "merchant category: Money Transfer"   instead of   "component V14"
"""
from __future__ import annotations

# --------------------------------------------------------------- dropped
# Every exclusion is deliberate and defensible out loud.
# MEASURED, not assumed. With a time-based split, the full 36-feature set
# scored val PR-AUC 0.00089 (1.5x baseline). Cutting to the features below took
# it to 0.04657 (76.9x) -- a 52x improvement -- because two whole families of
# feature were teaching the model things that do not survive the calendar:
#
#   IDENTITY FINGERPRINTS. There are only 2,000 cardholders and 6,146 cards in
#   this dataset, so (yearly_income, total_debt, credit_score, per_capita_income,
#   current_age, num_credit_cards) jointly identify a PERSON. The model learned
#   which people were defrauded in 2010-2016 instead of what fraud looks like,
#   and those people are not the ones defrauded in 2017+. This is also why a
#   RANDOM split flatters so badly: it scored 0.61 by memorising identities that
#   appear on both sides of the split.
#
#   CALENDAR-DRIFTING FEATURES. card_age_years and years_since_pin_change grow
#   monotonically with the date, so their distribution in test is one the model
#   never saw in train. Splits learned on them do not transfer.
#
DROPPED: dict[str, str] = {
    # --- identity fingerprints (2,000 users is small enough to memorise) ---
    "current_age": "identity: helps fingerprint one of only 2,000 cardholders.",
    "gender": "identity: part of the same cardholder fingerprint.",
    "per_capita_income": "identity: fingerprints the cardholder, not the behaviour.",
    "yearly_income": "identity: fingerprints the cardholder, not the behaviour.",
    "total_debt": "identity: fingerprints the cardholder, not the behaviour.",
    "credit_score": "identity: fingerprints the cardholder, not the behaviour.",
    "num_credit_cards": "identity: fingerprints the cardholder, not the behaviour.",
    "debt_to_income": "identity: derived from two cardholder fingerprint columns.",
    "amount_vs_card_median": "identity: keyed to one card's history.",
    "credit_limit": "identity: near-unique per card; the ratio to amount is kept.",
    "num_cards_issued": "identity: per-account constant.",
    "card_on_dark_web": "identity: per-card constant, and 'No' for almost every row.",
    # --- distribution marches with the calendar ---
    "card_age_years": "drifts with the calendar: systematically larger in later splits.",
    "years_since_pin_change": "drifts with the calendar, same failure mode.",
    "month": "encodes WHICH months were in train; does not transfer forward.",
    "day_of_month": "no fraud signal; adds splits the model can overfit.",
    "day_of_week": "no measurable lift once use_chip and MCC are present.",
    "is_weekend": "no measurable lift once use_chip and MCC are present.",
    "is_refund": "subsumed by abs_amount and the sign is rare among frauds.",
    "amount": "kept as abs_amount + is_online; the raw signed value adds nothing.",
    "amount_to_income": "identity: denominator is a cardholder fingerprint.",
    # Direct identifiers / PII. Never features.
    "card_number": "PII: card PAN. Never a feature.",
    "cvv": "PII: security code. Never a feature.",
    "address": "PII: cardholder street address.",
    "latitude": "PII: home coordinates; also not where the transaction happened.",
    "longitude": "PII: home coordinates; also not where the transaction happened.",
    # High-cardinality raw IDs. A tree can memorise these and score brilliantly
    # in-sample while learning nothing transferable. The semantic version of
    # 'which merchant' is the MCC category, which we DO keep.
    "client_id": "raw ID: memorisable, not generalisable (kept only for joins/splits).",
    "card_id": "raw ID: memorisable, not generalisable (kept only for joins).",
    "merchant_id": "raw ID: 100k+ levels; MCC category carries the transferable signal.",
    "merchant_city": "~14k levels; state + MCC carry the same signal without the sparsity.",
    "id": "transaction primary key.",
    # Redundant with a feature we already build.
    "birth_year": "redundant with current_age.",
    "birth_month": "redundant with current_age.",
    "expires": "redundant with card_age_years / card_expired flag.",
    "acct_open_date": "consumed into card_age_years.",
    "year_pin_last_changed": "consumed into years_since_pin_change.",
    "retirement_age": "not a fraud signal; correlates with age which we keep.",
    "date": "consumed into hour / day-of-week / month features.",
    "zip": "consumed into the is_online flag; raw ZIP is high-cardinality.",
}

# ----------------------------------------------------------- categoricals
# Passed as pandas `category` dtype. XGBoost (enable_categorical), CatBoost
# (cat_features) and sklearn HistGradientBoosting all consume these natively,
# so no one-hot explosion and no arbitrary ordinal encoding.
CATEGORICAL: list[str] = [
    "use_chip",
    "merchant_state",
    "mcc_category",
    "card_brand",
    "card_type",
    "error_kind",
]

# --------------------------------------------------------------- numeric
NUMERIC: list[str] = [
    "abs_amount", "log_amount",
    "amount_to_credit_limit", "amount_vs_mcc_median",
    "hour", "is_night",
    "is_online", "has_error", "has_chip",
]

FEATURES: list[str] = NUMERIC + CATEGORICAL
LABEL = "is_fraud"

# ------------------------------------------------- reviewer-facing labels
# What a human sees instead of a column name. This is the whole point of
# moving to a labelled dataset.
LABELS: dict[str, str] = {
    "amount": "transaction amount",
    "abs_amount": "transaction size (ignoring refund sign)",
    "log_amount": "transaction amount (log scale)",
    "is_refund": "is a refund (negative amount)",
    "hour": "hour of day",
    "day_of_week": "day of week",
    "day_of_month": "day of month",
    "month": "month of year",
    "is_night": "happened overnight (00:00-05:59)",
    "is_weekend": "happened at the weekend",
    "is_online": "online / card-not-present",
    "has_error": "the terminal reported an error",
    "error_kind": "terminal error type",
    "use_chip": "how the card was presented (chip / swipe / online)",
    "merchant_state": "merchant state",
    "mcc_category": "merchant category",
    "card_brand": "card brand",
    "card_type": "card type (credit / debit / prepaid)",
    "gender": "cardholder gender",
    "credit_limit": "card credit limit",
    "num_cards_issued": "cards issued on this account",
    "has_chip": "card has a chip",
    "card_on_dark_web": "card seen on the dark web",
    "card_age_years": "how long the account has been open (years)",
    "years_since_pin_change": "years since the PIN was last changed",
    "current_age": "cardholder age",
    "per_capita_income": "per-capita income in the cardholder's area",
    "yearly_income": "cardholder yearly income",
    "total_debt": "cardholder total debt",
    "credit_score": "cardholder credit score",
    "num_credit_cards": "credit cards held by the cardholder",
    "amount_to_credit_limit": "amount as a share of the card's credit limit",
    "debt_to_income": "cardholder debt-to-income ratio",
    "amount_to_income": "amount as a share of yearly income",
    "amount_vs_card_median": "amount vs this card's usual spend",
    "amount_vs_mcc_median": "amount vs typical spend in this merchant category",
}


def label(column: str) -> str:
    """Human-readable name for a feature. Falls back to the raw column."""
    return LABELS.get(column, column)
