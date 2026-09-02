"""Load, join, clean and split the labelled transaction dataset.

Four source tables are joined into one modelling frame:

    transactions_data.csv   13.3M rows, 8.9M of them labelled
    cards_data.csv          card attributes, joined on card_id
    users_data.csv          cardholder attributes, joined on client_id
    mcc_codes.json          merchant category code -> readable category name

Two rules are enforced loudly, because they are what separate a real result
from a hackathon result:

  1. THE SPLIT IS BY TIME, not at random. Fraud is a temporal problem: a model
     that trains on next month and predicts last month is scoring information
     it could never have had. The timeline runs train -> val -> test -> demo,
     so every evaluation looks forward, and the demo set is the most recent
     traffic in the dataset.

  2. AGGREGATES NEVER SEE THE FUTURE. There are two kinds here and they are
     protected differently:

     STATIC aggregates (the median spend per merchant category) are fitted on
     the TRAIN PERIOD ONLY and then applied unchanged to the later splits.
     Computing them over all time would embed the future in every row.

     ROLLING aggregates (this card's purchases in the last hour/day/week) are
     computed over the whole timeline, but every window is backward-looking
     (`closed="left"`, excluding the row being scored). That is not a leak: a
     transaction in the test period is entitled to know what its card did last
     week, because in production it would. It simply must never see anything
     that happened after itself.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, REPORTS
from src.schema import CATEGORICAL, FEATURES, LABEL, NUMERIC

IBM_RAW = DATA_RAW / "ibm"
# 90% of the data does the learning, 9% is the held-out test, 1% is a personal
# check set. The 90% is itself split: the models fit on `train`, while `val`
# pays for early stopping, ensemble weights, isotonic calibration and the
# operating threshold. Those four choices need data the models did not fit on
# -- making them on train would leak, and making them on test would spend the
# held-out split before it is reported.
SPLIT_FRACTIONS = {"train": 0.82, "val": 0.08, "test": 0.09, "demo": 0.01}

MONEY = ("amount", "credit_limit", "per_capita_income", "yearly_income", "total_debt")


def _money(s: pd.Series) -> pd.Series:
    """Parse a currency string like '$-77.00' into -77.00."""
    cleaned = (s.astype("string")
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")


def _yes_no(s: pd.Series) -> pd.Series:
    """'YES'/'No' -> 1/0. A missing value is treated as 'not yes' rather than
    propagating NA, so an incomplete card record cannot crash the pipeline."""
    return ((s.astype("string").str.strip().str.lower() == "yes")
            .fillna(False).astype("int8"))


# ------------------------------------------------------------------ loading
def load_labels() -> pd.Series:
    path = IBM_RAW / "train_fraud_labels.json"
    raw = json.loads(path.read_text(encoding="utf-8"))["target"]
    idx = np.fromiter(raw.keys(), dtype="int64", count=len(raw))
    val = np.fromiter((v == "Yes" for v in raw.values()), dtype="int8", count=len(raw))
    s = pd.Series(val, index=idx, name=LABEL)
    print(f"[prep] labels: {len(s):,} rows, {int(s.sum()):,} fraud ({s.mean():.4%})")
    return s


def load_transactions(labels: pd.Series) -> pd.DataFrame:
    """Stream the 1.26GB CSV, keeping only labelled rows."""
    path = IBM_RAW / "transactions_data.csv"
    usecols = ["id", "date", "client_id", "card_id", "amount", "use_chip",
               "merchant_state", "zip", "mcc", "errors"]
    dtype = {"id": "int64", "client_id": "int32", "card_id": "int32",
             "mcc": "int32", "amount": "string", "use_chip": "string",
             "merchant_state": "string", "errors": "string", "zip": "float32"}
    keep = set(labels.index.to_numpy().tolist())
    parts, seen, t0 = [], 0, time.time()
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=2_000_000, dtype=dtype):
        seen += len(chunk)
        parts.append(chunk[chunk["id"].isin(keep)])
        print(f"    scanned {seen:,} rows ...")
    df = pd.concat(parts, ignore_index=True)
    print(f"[prep] kept {len(df):,} labelled of {seen:,} transactions "
          f"in {time.time() - t0:.0f}s")
    return df


def join_context(df: pd.DataFrame) -> pd.DataFrame:
    cards = pd.read_csv(IBM_RAW / "cards_data.csv")
    users = pd.read_csv(IBM_RAW / "users_data.csv")
    mcc = json.loads((IBM_RAW / "mcc_codes.json").read_text(encoding="utf-8"))

    cards = cards.rename(columns={"id": "card_id"})[
        ["card_id", "card_brand", "card_type", "has_chip", "num_cards_issued",
         "credit_limit", "acct_open_date", "year_pin_last_changed",
         "card_on_dark_web"]]
    users = users.rename(columns={"id": "client_id"})[
        ["client_id", "current_age", "gender", "per_capita_income",
         "yearly_income", "total_debt", "credit_score", "num_credit_cards"]]

    df = df.merge(cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(users, on="client_id", how="left", validate="many_to_one")
    df["mcc_category"] = df["mcc"].astype("string").map(mcc).fillna("Unknown")
    print(f"[prep] joined cards ({len(cards):,}) and users ({len(users):,}); "
          f"{df['mcc_category'].nunique()} merchant categories")
    return df


# ------------------------------------------------------------------ cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    for c in MONEY:
        if c in df.columns:
            df[c] = _money(df[c]).astype("float32")
    df["has_chip"] = _yes_no(df["has_chip"])
    df["card_on_dark_web"] = _yes_no(df["card_on_dark_web"])

    # `errors` is 98.4% null -- absence is the norm, so presence is the signal.
    err = df["errors"].astype("string").fillna("")
    df["has_error"] = (err.str.len() > 0).astype("int8")
    kind = err.str.split(",").str[0]
    df["error_kind"] = kind.where(kind.str.len() > 0, "none").fillna("none")

    # A missing ZIP means card-not-present. The null IS the information, so we
    # encode it as a flag rather than imputing a fake ZIP.
    df["is_online"] = df["zip"].isna().astype("int8")
    df["merchant_state"] = df["merchant_state"].fillna("ONLINE")

    # Magnitude is computed here rather than in build_features because the
    # velocity windows run before the split and are denominated in it.
    df["abs_amount"] = df["amount"].abs().astype("float32")
    return df


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card behavioural history, computed causally.

    A transaction on its own says little. What a fraud analyst actually looks
    at is the CHANGE: this card normally spends 40 euros twice a week, and it
    has just made its fifth purchase in an hour. None of that is visible to a
    model that sees each row in isolation, and until now this one did.

    Two rules make these safe:

      1. EVERY WINDOW EXCLUDES THE CURRENT ROW (`closed="left"`). A count that
         includes itself is a constant 1 offset at best; at worst, on a sum, it
         leaks the very amount being scored.
      2. THEY ARE COMPUTED BEFORE THE TIME SPLIT, over the whole timeline. That
         is not leakage -- it is the opposite. A transaction in the test period
         is allowed to know what that card did last week, because in production
         it would. What it must never see is anything AFTER itself, which
         `closed="left"` on a time-ordered frame guarantees.

    Note these are behavioural, not identity: they describe what the card just
    did, not which card it is. That is the distinction that killed the static
    per-card median (see docs/FINDINGS.md) while keeping these.
    """
    df = df.sort_values(["card_id", "date"], kind="mergesort")
    g = df.set_index("date").groupby("card_id", observed=True)
    out = {}

    for window, tag in (("1h", "1h"), ("24h", "24h"), ("7d", "7d")):
        out[f"card_txns_{tag}"] = (
            g["abs_amount"].rolling(window, closed="left").count()
            .reset_index(level=0, drop=True))
    out["card_amount_24h"] = (
        g["abs_amount"].rolling("24h", closed="left").sum()
        .reset_index(level=0, drop=True))
    out["card_mean_amount_7d"] = (
        g["abs_amount"].rolling("7d", closed="left").mean()
        .reset_index(level=0, drop=True))

    for k, v in out.items():
        df[k] = v.to_numpy()

    # An EMPTY window is zero, not unknown. Pandas returns NaN when a rolling
    # window contains no rows, but "this card made no purchases in the last
    # hour" is a fact, and a very different one from "I have never seen this
    # card". Leaving both as NaN makes a quiet regular customer look identical
    # to a brand-new one and throws away the signal these features exist for.
    # Counts and sums therefore become 0; the genuinely-unknown case is carried
    # by hours_since_card_txn (NaN on a card's first ever transaction).
    for c in ("card_txns_1h", "card_txns_24h", "card_txns_7d", "card_amount_24h"):
        df[c] = df[c].fillna(0.0)
    # The MEAN of an empty window stays NaN -- there is no honest value for the
    # average of nothing, and the model handles NaN natively.

    # Time since the card's previous transaction. First-ever transaction on a
    # card has no predecessor: left as NaN rather than a fabricated zero, which
    # would read as "one second ago" -- the opposite of the truth.
    prev = df.groupby("card_id", observed=True)["date"].shift(1)
    df["hours_since_card_txn"] = (
        (df["date"] - prev).dt.total_seconds() / 3600.0).astype("float32")

    # How this amount compares to the card's own recent behaviour. This is the
    # "unusual for THIS customer" signal, and it generalises to cards never
    # seen because it is relative to their own history, not to an identity.
    denom = df["card_mean_amount_7d"].replace(0, np.nan)
    df["amount_vs_card_recent"] = (df["abs_amount"] / denom).astype("float32")

    # First time this card has transacted in this merchant category.
    seen_before = df.groupby(["card_id", "mcc_category"], observed=True).cumcount()
    df["new_mcc_for_card"] = (seen_before == 0).astype("int8")

    for c in ("card_txns_1h", "card_txns_24h", "card_txns_7d",
              "card_amount_24h", "card_mean_amount_7d"):
        df[c] = df[c].astype("float32")

    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    print(f"[prep] velocity features built over {len(df):,} rows "
          f"({df['card_id'].nunique():,} cards); windows exclude the current row")
    return df


def build_features(df: pd.DataFrame, stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Feature engineering.

    `stats` are train-period aggregates. When None they are learned from `df`,
    which only ever happens on the train split.
    """
    d = df["date"].dt
    df["hour"] = d.hour.astype("int8")
    df["day_of_week"] = d.dayofweek.astype("int8")
    df["day_of_month"] = d.day.astype("int8")
    df["month"] = d.month.astype("int8")
    df["is_night"] = (df["hour"] < 6).astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    df["abs_amount"] = df["amount"].abs().astype("float32")
    df["is_refund"] = (df["amount"] < 0).astype("int8")
    df["log_amount"] = np.log1p(df["abs_amount"]).astype("float32")

    open_date = pd.to_datetime(df["acct_open_date"], format="%m/%Y", errors="coerce")
    df["card_age_years"] = ((df["date"] - open_date).dt.days / 365.25).astype("float32")
    df["years_since_pin_change"] = (
        d.year - pd.to_numeric(df["year_pin_last_changed"], errors="coerce")
    ).astype("float32")

    lim = df["credit_limit"].replace(0, np.nan)
    inc = df["yearly_income"].replace(0, np.nan)
    df["amount_to_credit_limit"] = (df["abs_amount"] / lim).astype("float32")
    df["amount_to_income"] = (df["abs_amount"] / inc).astype("float32")
    df["debt_to_income"] = (df["total_debt"] / inc).astype("float32")

    # ---- train-period aggregates: the leakage-sensitive part ----
    if stats is None:
        stats = {
            "card_median": df.groupby("card_id")["abs_amount"].median().to_dict(),
            "mcc_median": df.groupby("mcc_category", observed=True)["abs_amount"]
                            .median().to_dict(),
            "global_median": float(df["abs_amount"].median()),
        }
    g = stats["global_median"]
    card_med = df["card_id"].map(stats["card_median"]).fillna(g).astype("float32")
    mcc_med = df["mcc_category"].map(stats["mcc_median"]).fillna(g).astype("float32")
    df["amount_vs_card_median"] = (
        df["abs_amount"] / card_med.replace(0, np.nan)).astype("float32")
    df["amount_vs_mcc_median"] = (
        df["abs_amount"] / mcc_med.replace(0, np.nan)).astype("float32")

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    for c in NUMERIC:
        if df[c].dtype.kind not in "fi":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].astype("float32")
    return df, stats


# -------------------------------------------------------------------- split
def time_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Chronological train -> val -> test -> demo. Every split looks FORWARD."""
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    n = len(df)
    order = ["train", "val", "test", "demo"]
    parts, start = {}, 0
    for i, name in enumerate(order):
        end = n if i == len(order) - 1 else start + int(n * SPLIT_FRACTIONS[name])
        parts[name] = df.iloc[start:end]
        start = end
    print("[prep] time-based split (evaluation always looks FORWARD):")
    for k, p in parts.items():
        print(f"    {k:<6} {len(p):>10,} rows  ({len(p) / n:5.1%})  "
              f"{p['date'].min().date()} -> {p['date'].max().date()}  "
              f"fraud {int(p[LABEL].sum()):>5,} ({p[LABEL].mean():.3%})")
    return parts


def assert_no_leakage(parts: dict[str, pd.DataFrame]) -> None:
    """Loud and explicit: checked by transaction id AND by time boundary."""
    ids = {k: set(p["id"].to_numpy().tolist()) for k, p in parts.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[a] & ids[b]
        assert not overlap, f"LEAKAGE: {len(overlap)} ids in both {a} and {b}"
    assert parts["train"]["date"].max() <= parts["val"]["date"].min(), \
        "LEAKAGE: train extends past the start of val"
    assert parts["val"]["date"].max() <= parts["test"]["date"].min(), \
        "LEAKAGE: val extends past the start of test"
    total = sum(len(p) for p in parts.values())
    assert len(set().union(*ids.values())) == total, "duplicate ids across splits"
    print(f"[prep] LEAKAGE CHECK PASSED: {total:,} unique ids, no id in two "
          f"splits, splits strictly ordered in time")


def main() -> None:
    labels = load_labels()
    df = load_transactions(labels)
    df[LABEL] = df["id"].map(labels).astype("int8")
    df = join_context(df)
    df = clean(df)
    # Velocity is computed over the WHOLE timeline, before the split. That is
    # not leakage: each window looks only backwards (closed="left"), so a test
    # transaction may use its card's earlier history exactly as production
    # would, and can never see anything after itself.
    df = add_velocity(df)

    parts = time_split(df)
    del df

    train, stats = build_features(parts["train"].copy(), None)
    out = {"train": train}
    for name in ("val", "test", "demo"):
        out[name], _ = build_features(parts[name].copy(), stats)
    assert_no_leakage(out)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cols = FEATURES + [LABEL, "id", "date"]
    for name, part in out.items():
        path = DATA_PROCESSED / f"{name}.parquet"
        part[cols].to_parquet(path, index=False)
        print(f"[prep] wrote {path}  {part[cols].shape}")

    # Into models/, not data/processed/: the API needs these at serving time
    # and data/processed is gitignored.
    from src.config import MODELS

    with open(MODELS / "agg_stats.json", "w", encoding="utf-8") as fh:
        json.dump({"global_median": stats["global_median"],
                   "mcc_median": stats["mcc_median"]}, fh)

    summary = {
        "rows_labelled": int(sum(len(p) for p in out.values())),
        "fraud_total": int(sum(int(p[LABEL].sum()) for p in out.values())),
        "splits": {k: {"rows": int(len(p)), "fraud": int(p[LABEL].sum()),
                       "fraud_rate": float(p[LABEL].mean()),
                       "start": str(p["date"].min()), "end": str(p["date"].max())}
                   for k, p in out.items()},
        "n_features": len(FEATURES),
        "split_strategy": "time-based 70/15/15, aggregates fitted on train only",
    }
    (REPORTS / "split_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print("[prep] wrote reports/split_summary.json")


if __name__ == "__main__":
    main()
