"""Out-of-fold scoring and probability calibration.

TWO PROBLEMS, ONE MECHANISM
---------------------------
1. THE THRESHOLD WAS SELECTED IN-SAMPLE. `src.tune_optuna.refit_final` refits
   the winning model on train+val so no data is wasted -- correct -- but
   `src.evaluate` then picks the operating threshold as the argmin of the
   validation cost curve. For a model that has already trained on validation,
   that curve is 0 everywhere below 0.50 (it separates its own training data
   perfectly), so `np.argmin` returns the first index of a tie and the shipped
   threshold of 0.09 is an artefact of tie-breaking, not a decision. Measured:
   on val the tuned model shows tp=71 fp=0 fn=0.

2. RESAMPLING BREAKS CALIBRATION. Train on a 5% fraud rate and the model
   outputs probabilities for a world with a 5% fraud rate, not the 0.17% one
   the merchant lives in. The ranking survives, the numbers do not -- and the
   cost model multiplies those numbers by money.

Both are fixed by the same thing: scores for rows the model did not train on.
`oof_predictions` runs stratified K-fold over train+val and refits the whole
recipe -- resampling, weighting, custom loss and all -- inside each fold, so
every score is out-of-sample. That yields ~400 frauds' worth of honest scores
to calibrate on and to place the threshold on, against the 71 a single
validation split offers.

RESAMPLING INSIDE THE FOLD IS THE WHOLE POINT. Oversample first and split
afterwards and a synthetic fraud interpolated from row 7 lands in the training
fold while row 7 itself lands in the held-out fold. The model then scores a
near-copy of something it has seen. It is the most common way SMOTE produces
99% recall that evaporates in production.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from src.imbalance import FocalLoss, resample, sample_weights, sigmoid, smooth_labels


# ------------------------------------------------------------- recipe -> model
def fit_recipe(X_tr: np.ndarray, y_tr: np.ndarray, amt_tr: np.ndarray, recipe: dict,
               feature_names: list[str], seed: int,
               X_es=None, y_es=None) -> tuple[xgb.Booster, bool]:
    """Train one model under one imbalance recipe. Returns (booster, is_margin).

    `is_margin` is True when a custom objective was used, because XGBoost then
    returns raw margins from `predict` and the caller has to apply the sigmoid
    itself. Getting that wrong silently produces scores outside [0, 1] that the
    cost model happily thresholds at 0.5 -- so it is returned, not assumed.
    """
    Xr, yr = resample(recipe.get("resampler", "none"), X_tr, y_tr,
                      ratio=recipe.get("ratio", 0.0), seed=seed, split="train")

    # Sample weights are computed AFTER resampling so a synthetic fraud is
    # weighted like a fraud. Amount-based weighting needs an amount for the
    # synthetic rows: they inherit the median fraud amount rather than an
    # interpolated one, because interpolating money across two unrelated
    # transactions invents a price nobody paid.
    n_new = len(yr) - len(y_tr)
    if n_new > 0:
        fraud_amt = amt_tr[y_tr == 1]
        fill = float(np.median(fraud_amt)) if len(fraud_amt) else 0.0
        amt_r = np.concatenate([amt_tr, np.full(n_new, fill)])
    else:
        amt_r = amt_tr[: len(yr)] if len(yr) < len(y_tr) else amt_tr

    w = sample_weights(recipe.get("weighting", "none"), yr,
                       amounts=amt_r, fp_cost=recipe.get("fp_cost", 50.0))

    labels = smooth_labels(yr, eps_pos=recipe.get("eps_pos", 0.0),
                           eps_neg=recipe.get("eps_neg", 0.0))

    params = dict(recipe["params"])
    params["seed"] = seed
    if recipe.get("scale_pos_weight") == "auto":
        params["scale_pos_weight"] = float((yr == 0).sum() / max((yr == 1).sum(), 1))
    elif recipe.get("scale_pos_weight") is not None:
        params["scale_pos_weight"] = float(recipe["scale_pos_weight"])
    else:
        params.pop("scale_pos_weight", None)

    obj = None
    if recipe.get("focal"):
        obj = FocalLoss(**recipe["focal"])
        params.pop("objective", None)
        params.pop("eval_metric", None)
        params["base_score"] = 0.5
        params["disable_default_eval_metric"] = 1

    dtrain = xgb.DMatrix(Xr, label=labels, weight=w, feature_names=feature_names)
    evals, es = [], None
    if X_es is not None:
        evals = [(xgb.DMatrix(X_es, label=y_es, feature_names=feature_names), "es")]
        es = recipe.get("early_stopping", 40)
    if obj is not None:
        evals, es = [], None  # custom metric plumbing adds noise; use fixed rounds

    booster = xgb.train(params, dtrain, num_boost_round=recipe["rounds"],
                        evals=evals, early_stopping_rounds=es, verbose_eval=False,
                        obj=obj)
    return booster, obj is not None


def score_with(booster: xgb.Booster, X: np.ndarray, feature_names: list[str],
               is_margin: bool) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=feature_names)
    p = booster.predict(d, output_margin=is_margin)
    return sigmoid(p) if is_margin else p


# ------------------------------------------------------------------ out-of-fold
def oof_predictions(X: np.ndarray, y: np.ndarray, amounts: np.ndarray, recipe: dict,
                    feature_names: list[str], *, n_splits: int = 5, seed: int = 42,
                    verbose: bool = True) -> np.ndarray:
    """Stratified K-fold out-of-fold scores under the full recipe.

    Every fold refits from scratch, including the resampling, so no synthetic
    row ever derives from a transaction being scored in the held-out fold.
    """
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(skf.split(X, y), 1):
        booster, is_margin = fit_recipe(X[tr], y[tr], amounts[tr], recipe,
                                        feature_names, seed + i)
        oof[va] = score_with(booster, X[va], feature_names, is_margin)
        if verbose:
            print(f"  [oof] fold {i}/{n_splits}  train={len(tr):,} "
                  f"held-out={len(va):,} frauds_held_out={int(y[va].sum())}")
    return oof


# ----------------------------------------------------------------- calibration
class Calibrator:
    """Isotonic or Platt mapping from raw score to calibrated probability.

    Isotonic is monotone and non-parametric, so it cannot reorder transactions
    -- PR-AUC and ROC-AUC are unchanged by construction and only the numbers
    the cost model consumes move. Platt is a single sigmoid: less flexible,
    but it cannot produce the flat steps isotonic creates where data is thin,
    which is most of the interesting range here.
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.model = None
        self._knots: tuple[np.ndarray, np.ndarray] | None = None
        self._platt: tuple[float, float] | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(p, y)
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression

            z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
            self.model = LogisticRegression(C=1e6, solver="lbfgs")
            self.model.fit(z.reshape(-1, 1), y)
        elif self.method == "none":
            self.model = None
        else:
            raise KeyError(f"unknown calibration method {self.method!r}")
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none" or self.model is None:
            return p
        if self.method == "isotonic":
            if self._knots is not None:
                return np.clip(np.interp(p, self._knots[0], self._knots[1]), 0.0, 1.0)
            return np.clip(self.model.predict(p), 0.0, 1.0)
        z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
        if self._platt is not None:
            a, b = self._platt
            return 1.0 / (1.0 + np.exp(-np.clip(a * z + b, -50, 50)))
        return self.model.predict_proba(z.reshape(-1, 1))[:, 1]

    # -- persistence ------------------------------------------------------
    # Stored as plain numbers, not a pickle. The API loads this at startup and
    # a pickle would mean the serving path executes whatever the artefact says;
    # isotonic is a step function and Platt is two floats, so neither needs one.
    def to_dict(self) -> dict:
        if self.method == "isotonic":
            f = self.model.f_
            return {"method": "isotonic", "x": list(map(float, f.x)),
                    "y": list(map(float, f.y))}
        if self.method == "platt":
            return {"method": "platt", "a": float(self.model.coef_[0][0]),
                    "b": float(self.model.intercept_[0])}
        return {"method": "none"}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        c = cls(d.get("method", "none"))
        if c.method == "isotonic":
            c._knots = (np.asarray(d["x"], dtype=float), np.asarray(d["y"], dtype=float))
            c.model = True  # sentinel: fitted, served from knots
        elif c.method == "platt":
            c._platt = (float(d["a"]), float(d["b"]))
            c.model = True
        return c


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def expected_calibration_error(y, p, n_bins: int = 15, strategy: str = "quantile") -> float:
    """ECE over quantile bins.

    Quantile bins, not equal-width: at a 0.17% base rate an equal-width binning
    puts 99.9% of transactions in the first bin and reports a flattering number
    computed almost entirely from rows scored near zero.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        return 0.0
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def cross_fitted_calibration(y, p, method: str, *, n_splits: int = 5,
                             seed: int = 42) -> dict:
    """Honest calibration metrics.

    Fitting isotonic on a set and then reporting its ECE on that same set gives
    ~0, because isotonic's whole job is to make the bin means match the observed
    rates of the data it was shown. That number is a tautology, not evidence.
    So the calibrator is refit on 4/5 of the scores and scored on the held-out
    fifth, and it is those out-of-fold numbers that go in the report.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    out = np.zeros(len(p), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(p.reshape(-1, 1), y):
        out[te] = Calibrator(method).fit(p[tr], y[tr]).predict(p[te])
    return {"method": method, "brier": brier(y, out),
            "ece": expected_calibration_error(y, out),
            "mean_score": float(out.mean()), "n_splits": n_splits}


def calibration_report(y, p_raw, p_cal, method: str) -> dict:
    return {
        "method": method,
        "brier_raw": brier(y, p_raw),
        "brier_calibrated": brier(y, p_cal),
        "ece_raw": expected_calibration_error(y, p_raw),
        "ece_calibrated": expected_calibration_error(y, p_cal),
        "mean_score_raw": float(np.mean(p_raw)),
        "mean_score_calibrated": float(np.mean(p_cal)),
        "observed_base_rate": float(np.mean(y)),
    }


def plot_reliability(y, p_raw, p_cal, out, n_bins: int = 12) -> None:
    """Reliability diagram on quantile bins, log-log because everything
    interesting on this dataset happens below p = 0.01."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for p, label, colour in ((p_raw, "raw score", "#dc2626"),
                             (p_cal, "calibrated", "#2563eb")):
        p = np.asarray(p, dtype=float)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p <= hi)
            if m.sum() < 20:
                continue
            xs.append(max(p[m].mean(), 1e-6))
            ys.append(max(np.asarray(y, dtype=float)[m].mean(), 1e-6))
        ax.plot(xs, ys, "o-", color=colour, label=label, ms=4, lw=1.4)
    lim = [1e-6, 1.0]
    ax.plot(lim, lim, ls="--", lw=1, color="#94a3b8", label="perfect calibration")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean predicted probability (quantile bin)")
    ax.set_ylabel("observed fraud rate in bin")
    ax.set_title("Reliability - out-of-fold scores")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[calib] wrote {out}")
