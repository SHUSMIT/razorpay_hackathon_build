"""Reviewer-facing evidence for every feature the model used.

A SHAP value tells you a feature mattered. It does not tell a human WHY, and a
fraud analyst cannot act on "amount_vs_mcc_median contributed +1.4".

This module supplies the missing half, and every number in it is measured on
the TRAIN split only -- never invented, never taken from the held-out data:

  NUMERIC features get a position:
      "amount 142.50 - higher than 99.2% of legitimate transactions"

  CATEGORICAL features get a base rate:
      "merchant category 'Money Transfer' - 2.1% of transactions in this
       category were fraud, against 0.16% overall (13x)"

  The SCORE gets an empirical fraud rate for its band, so a reviewer knows what
  a 0.4 actually means in outcomes rather than in model units.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import MODELS
from src.data import load_split
from src.weights import recency_weights
from src.schema import CATEGORICAL, FEATURES, LABEL, NUMERIC, label

CONTEXT_FILE = MODELS / "feature_context.json"
QUANTILE_GRID = list(range(0, 101))
SCORE_BAND_EDGES = [0.0, 0.001, 0.005, 0.02, 0.05, 0.10, 0.25, 0.50, 1.01]
MIN_CATEGORY_SUPPORT = 200

# Yes/no features. A percentile is meaningless for these -- "higher than 100%
# of legitimate transactions" is not a sentence a reviewer can use -- so they
# get a fraud rate like a category instead.
BINARY = ("is_online", "is_night", "has_error", "has_chip")


def format_value(name: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not provided"
    if name in ("abs_amount",):
        return f"{value:,.2f}"
    if name == "log_amount":
        return f"{float(np.expm1(value)):,.2f}"
    if name == "hour":
        return f"{int(value):02d}:00"
    if name in ("is_online", "is_night", "has_error", "has_chip"):
        return "yes" if float(value) >= 0.5 else "no"
    if isinstance(value, str):
        return value
    return f"{value:,.4g}"


class FeatureContext:
    def __init__(self, payload: dict | None = None):
        payload = payload or {}
        self.knots: dict[str, list[float]] = payload.get("knots", {})
        self.category_rates: dict[str, dict] = payload.get("category_rates", {})
        self.bands: list[dict] = payload.get("score_bands", {}).get("bands", [])
        self.band_source = payload.get("score_bands", {}).get("source", "")
        self.n_reference = int(payload.get("n_reference_rows", 0))
        self.overall_rate = float(payload.get("overall_fraud_rate", 0.0))
        self.available = bool(self.knots or self.category_rates)

    @classmethod
    def load(cls) -> "FeatureContext":
        if CONTEXT_FILE.exists():
            try:
                return cls(json.loads(CONTEXT_FILE.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
        return cls({})

    # ------------------------------------------------------------ numeric
    def percentile(self, name: str, value: float) -> float | None:
        knots = self.knots.get(name)
        if not knots or value is None or not np.isfinite(value):
            return None
        arr = np.asarray(knots, dtype="float64")
        if value <= arr[0]:
            return 0.0
        if value >= arr[-1]:
            return 1.0
        return float(np.interp(value, arr, np.asarray(QUANTILE_GRID) / 100.0))

    # -------------------------------------------------------------- both
    def describe(self, name: str, value) -> dict:
        """Human-readable evidence for one feature value."""
        shown = format_value(name, value)
        friendly = label(name)
        out = {"feature": name, "label": friendly, "value_str": shown,
               "percentile": None, "lift": None}

        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            out["evidence"] = f"{friendly}: not provided"
            return out

        if name in CATEGORICAL or name in BINARY:
            key = shown if name in BINARY else str(value)
            stats = self.category_rates.get(name, {}).get(key)
            if not stats:
                out["evidence"] = f"{friendly}: {shown} (no reference data)"
                return out
            rate, n = stats["fraud_rate"], stats["n"]
            lift = (rate / self.overall_rate) if self.overall_rate else None
            out["lift"] = round(lift, 2) if lift else None
            tail = f" ({lift:.1f}x the overall rate)" if lift and lift >= 1.5 else ""
            out["evidence"] = (f"{friendly}: {shown} — {rate:.2%} of transactions "
                               f"in this group were fraud{tail}, against "
                               f"{self.overall_rate:.2%} overall (n={n:,})")
            return out

        pct = self.percentile(name, float(value))
        if pct is None:
            out["evidence"] = f"{friendly}: {shown}"
            return out
        higher = pct >= 0.5
        side = (pct if higher else 1.0 - pct) * 100.0
        out["percentile"] = round(pct, 6)
        out["evidence"] = (f"{friendly}: {shown} — {'higher' if higher else 'lower'} "
                           f"than {side:.1f}% of legitimate transactions")
        return out

    def band_fraud_rate(self, score: float) -> dict | None:
        for b in self.bands:
            if b["lo"] <= score < b["hi"]:
                return b
        return self.bands[-1] if self.bands else None


# ------------------------------------------------------------------ build
def _shipped_half_life() -> int:
    """The recency half-life the served model was actually trained with."""
    from src.config import SERVING_CONFIG

    try:
        shipped = json.loads(SERVING_CONFIG.read_text(encoding="utf-8")).get(
            "shipped_model")
        names = (["xgboost", "catboost", "histgb"] if shipped == "ensemble"
                 else [shipped])
        for n in names:
            path = MODELS / f"best_params_{n}.json"
            if path.exists():
                hl = json.loads(path.read_text(encoding="utf-8"))[
                    "best_params"].get("recency_half_life_days")
                if hl:
                    return int(hl)
    except Exception:  # noqa: BLE001 - context must build without a config
        pass
    return 0


def build(val_scores: np.ndarray | None = None,
          val_labels: np.ndarray | None = None) -> dict:
    """Learn the reference distributions from TRAIN only.

    `val_scores`/`val_labels` are optional: when the trained model's validation
    predictions are passed in, the score-band base rates are computed from them.
    Validation is genuinely out-of-sample here -- models are fit on train alone
    and validation is used only for early stopping and blend weights.
    """
    train = load_split("train", columns=FEATURES + [LABEL, "date"])
    y = train[LABEL].to_numpy()

    # The base rates a reviewer is shown MUST come from the same regime the
    # model was fitted on. The model is recency-weighted, so unweighted
    # whole-history rates actively contradict it: chip transactions look
    # low-risk across 2010-2018, while the model -- correctly, for the recent
    # period -- treats them as the dominant attack. Showing a reviewer "0.03%
    # of these were fraud" next to "this increased risk" destroys their trust
    # in the explanation, and the explanation is the point.
    half_life = _shipped_half_life()
    w = recency_weights(train["date"], half_life)
    if half_life:
        print(f"[context] base rates weighted to match the model's "
              f"{half_life}-day recency half-life")
    overall = float(np.average(y, weights=w))
    legit = train[train[LABEL] == 0]
    print(f"[context] reference = {len(legit):,} legitimate TRAIN rows; "
          f"overall fraud rate {overall:.4%}")

    knots = {}
    for c in NUMERIC:
        col = legit[c].to_numpy(dtype="float64")
        col = col[np.isfinite(col)]
        if len(col):
            knots[c] = np.percentile(col, QUANTILE_GRID).round(8).tolist()

    category_rates: dict[str, dict] = {}
    frame = train[list(CATEGORICAL)].copy()
    for c in BINARY:
        frame[c] = np.where(train[c].to_numpy(dtype="float64") >= 0.5, "yes", "no")

    for c in list(CATEGORICAL) + list(BINARY):
        g = pd.DataFrame({"level": frame[c].astype("object"), "y": y, "w": w})
        agg = g.groupby("level", observed=True).apply(
            lambda d: pd.Series({
                "fraud_rate": float(np.average(d["y"], weights=d["w"]))
                if d["w"].sum() > 0 else float(d["y"].mean()),
                "size": int(len(d)),
            }), include_groups=False)
        agg = agg[agg["size"] >= MIN_CATEGORY_SUPPORT]
        category_rates[c] = {
            str(k): {"fraud_rate": float(r["fraud_rate"]), "n": int(r["size"])}
            for k, r in agg.iterrows()
        }
        if len(agg):
            top = agg.sort_values("fraud_rate", ascending=False).head(3)
            print(f"[context] {c}: highest-risk levels -> " + ", ".join(
                f"{k} {r['fraud_rate']:.2%}" for k, r in top.iterrows()))

    payload = {
        "source": "train split",
        "n_reference_rows": int(len(legit)),
        "overall_fraud_rate": overall,
        "quantile_grid": QUANTILE_GRID,
        "knots": knots,
        "category_rates": category_rates,
        "score_bands": _score_bands(val_scores, val_labels),
    }
    CONTEXT_FILE.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[context] wrote {CONTEXT_FILE}")
    return payload


def _score_bands(scores, labels) -> dict:
    if scores is None or labels is None:
        print("[context] no validation predictions supplied -- score bands skipped")
        return {"source": "unavailable", "bands": []}
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    bands = []
    for lo, hi in zip(SCORE_BAND_EDGES[:-1], SCORE_BAND_EDGES[1:]):
        m = (scores >= lo) & (scores < hi)
        n = int(m.sum())
        f = int(labels[m].sum())
        bands.append({"lo": float(lo), "hi": float(hi), "n": n, "frauds": f,
                      "fraud_rate": (f / n) if n else None})
    print("[context] fraud rate by score band (validation, out-of-sample)")
    for b in bands:
        rate = "n/a" if b["fraud_rate"] is None else f"{b['fraud_rate']:.2%}"
        print(f"    [{b['lo']:.3f}, {b['hi']:.3f})  n={b['n']:>9,}  "
              f"frauds={b['frauds']:>5}  rate={rate}")
    return {"source": "validation split (out-of-sample)", "bands": bands}


def shipped_val_scores():
    """Validation scores from the model that is actually being served.

    Reproduces the serving path exactly -- same blend weights, same calibrator
    -- so the base rates a reviewer is shown correspond to the numbers the API
    actually emits. Uses the checkpointed validation predictions, so this costs
    nothing to recompute.
    """
    import numpy as np

    from src.calibration import Calibration
    from src.config import SERVING_CONFIG, VAL_PRED_DIR
    from src.data import load_categories, xy
    from src.models import ensemble_predict

    if not SERVING_CONFIG.exists():
        return None, None
    cfg = json.loads(SERVING_CONFIG.read_text(encoding="utf-8"))
    shipped = cfg.get("shipped_model")
    weights = cfg.get("ensemble_weights", {}) or {}

    preds = {p.stem: np.load(p) for p in sorted(VAL_PRED_DIR.glob("*.npy"))}
    if not preds:
        return None, None
    if shipped == "ensemble":
        members = {k: v for k, v in preds.items() if weights.get(k, 0) > 0}
        scores = ensemble_predict(members or preds, weights)
    elif shipped in preds:
        scores = preds[shipped]
    else:
        return None, None

    scores = Calibration.load().predict(scores)
    _, y = xy("val", load_categories())
    return scores, y.to_numpy()


if __name__ == "__main__":
    s, y = shipped_val_scores()
    if s is None:
        print("[context] no serving config or validation predictions found -- "
              "run `python run.py assess` first; score bands will be skipped")
    build(s, y)
