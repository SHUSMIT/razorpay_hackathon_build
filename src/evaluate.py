"""Held-out evaluation, threshold sweep and the explicit money cost model.

Nothing in here ever looks at the test split except `evaluate_model`, which is
called once, at the very end of a run.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from src.config import (
    BASELINE_MODEL,
    CHARGEBACK_FEE,
    FN_COST_MODE,
    FP_COST,
    REPORTS,
    REVIEW_BAND_WIDTH,
    REVIEW_CAPACITY_RATE,
    SERVING_CONFIG,
    TUNED_MODEL,
)
from src.train import xy


def _review_threshold(block_threshold: float) -> float:
    """Lower edge of the human-review band.

    Preferred source is review CAPACITY: the score above which we would send
    REVIEW_CAPACITY_RATE of traffic to a human, read off out-of-fold
    predictions (never the test split). Falls back to the old
    block_threshold * REVIEW_BAND_WIDTH rule when the context artefact has not
    been built yet.
    """
    try:
        from src.context import FeatureContext

        t = FeatureContext.load().threshold_for_capacity(REVIEW_CAPACITY_RATE)
    except Exception:  # noqa: BLE001 - context is optional
        t = None

    fallback = round(block_threshold * REVIEW_BAND_WIDTH, 4)
    if t is None or not (0.0 < t < block_threshold):
        print(f"[eval] review band from fallback rule: >= {fallback:.4f}")
        return fallback
    t = round(float(t), 4)
    print(f"[eval] review band sized by capacity "
          f"({REVIEW_CAPACITY_RATE:.2%} of traffic): >= {t:.4f}")
    return t


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def predict(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=list(X.columns))
    return booster.predict(d)


def core_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ------------------------------------------------------------------ cost model
def cost_at(y: np.ndarray, p: np.ndarray, amounts: np.ndarray, threshold: float) -> dict:
    """Money lost at one decision threshold.

    FN  -> we let fraud through: we eat the transaction amount (+ any fee).
    FP  -> we block a good customer: we eat FP_COST of support/goodwill.
    TP / TN cost nothing in this model; TP is where the saving shows up.
    """
    pred = p >= threshold
    is_fraud = y == 1
    fn_mask = is_fraud & ~pred
    fp_mask = (~is_fraud) & pred

    if FN_COST_MODE == "amount":
        fn_cost = float(amounts[fn_mask].sum()) + CHARGEBACK_FEE * int(fn_mask.sum())
    else:
        avg = float(amounts[is_fraud].mean()) if is_fraud.any() else 0.0
        fn_cost = (avg + CHARGEBACK_FEE) * int(fn_mask.sum())
    fp_cost = FP_COST * int(fp_mask.sum())

    tp, fp_n, fn_n = int((is_fraud & pred).sum()), int(fp_mask.sum()), int(fn_mask.sum())
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp_n,
        "fn": fn_n,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "total_cost": fn_cost + fp_cost,
        "precision": tp / (tp + fp_n) if tp + fp_n else 0.0,
        "recall": tp / (tp + fn_n) if tp + fn_n else 0.0,
        "block_rate": float(pred.mean()),
    }


def bootstrap_pr_auc(y: np.ndarray, p: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Bootstrap confidence interval for PR-AUC.

    The test split contains only 71 frauds. A PR-AUC quoted to four decimal
    places off 71 positives implies a precision the data cannot support, so we
    report the interval alongside the point estimate. Resampling is stratified
    within each class to keep the positive count fixed -- otherwise the
    resampled base rate moves and PR-AUC is not comparable across draws.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        scores[i] = average_precision_score(y[idx], p[idx])
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {"point": float(average_precision_score(y, p)),
            "ci_low": float(lo), "ci_high": float(hi),
            "std": float(scores.std()), "n_boot": n_boot}


def paired_bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                           n_boot: int = 2000, seed: int = 0) -> dict:
    """Is model B's PR-AUC really different from model A's?

    Paired: both models are scored on the SAME resampled rows each draw, which
    removes the split's own sampling noise and isolates the difference between
    the models. If the interval straddles zero, the two are indistinguishable
    on this much data and claiming an improvement would be dishonest.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        deltas[i] = (average_precision_score(y[idx], p_b[idx])
                     - average_precision_score(y[idx], p_a[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_point": float(average_precision_score(y, p_b)
                                 - average_precision_score(y, p_a)),
            "ci_low": float(lo), "ci_high": float(hi),
            "p_b_better_fraction": float((deltas > 0).mean()),
            "significant_at_95": bool(lo > 0 or hi < 0), "n_boot": n_boot}


def sweep_thresholds(y, p, amounts, lo=0.01, hi=0.99, step=0.01) -> pd.DataFrame:
    grid = np.round(np.arange(lo, hi + 1e-9, step), 4)
    return pd.DataFrame([cost_at(y, p, amounts, t) for t in grid])


def do_nothing_cost(y, amounts) -> float:
    """Baseline for comparison: block nothing, eat every fraud."""
    return float(amounts[y == 1].sum() + CHARGEBACK_FEE * int((y == 1).sum()))


# --------------------------------------------------------------------- plots
def plot_pr_curve(y, p, out, title):
    plt = _mpl()
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    base = float(np.mean(y))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rec, prec, lw=2, color="#2563eb", label=f"PR-AUC = {ap:.4f}")
    ax.axhline(base, ls="--", lw=1, color="#94a3b8", label=f"random baseline = {base:.5f}")
    ax.set_xlabel("Recall (fraud caught)")
    ax.set_ylabel("Precision (blocks that were really fraud)")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[eval] wrote {out}")


def plot_cost_curve(sweep: pd.DataFrame, best: dict, nothing: float, out):
    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax.plot(sweep.threshold, sweep.total_cost, lw=2, color="#dc2626", label="total cost")
    ax.plot(sweep.threshold, sweep.fn_cost, lw=1.2, ls="--", color="#b45309",
            label="missed-fraud cost (FN)")
    ax.plot(sweep.threshold, sweep.fp_cost, lw=1.2, ls="--", color="#0891b2",
            label=f"friction cost (FP @ {FP_COST:.0f}/block)")
    ax.axhline(nothing, ls=":", lw=1.2, color="#64748b", label=f"do nothing = {nothing:,.0f}")
    ax.axvline(best["threshold"], color="#16a34a", lw=1.5)
    ax.annotate(
        f"cost-optimal t = {best['threshold']:.2f}\ncost = {best['total_cost']:,.0f}"
        f"\nprecision {best['precision']:.3f} / recall {best['recall']:.3f}",
        xy=(best["threshold"], best["total_cost"]),
        xytext=(0.45, 0.72), textcoords="axes fraction", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#16a34a"),
    )
    ax.set_ylabel("cost on the held-out test split (currency units)")
    ax.set_title("Cost vs decision threshold (held-out test split)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2.plot(sweep.threshold, sweep.precision, label="precision", color="#2563eb")
    ax2.plot(sweep.threshold, sweep.recall, label="recall", color="#7c3aed")
    ax2.axvline(best["threshold"], color="#16a34a", lw=1.5)
    ax2.set_xlabel("decision threshold")
    ax2.set_ylabel("rate")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[eval] wrote {out}")


# ------------------------------------------------------------------ entrypoint
def evaluate_model(model_path=TUNED_MODEL, tag="tuned", split="test") -> dict:
    booster = xgb.Booster()
    booster.load_model(model_path)

    X, y_s = xy(split)
    y = y_s.to_numpy()
    amounts = X["Amount"].to_numpy()
    p = predict(booster, X)

    metrics = core_metrics(y, p, 0.5)
    ci = bootstrap_pr_auc(y, p)
    metrics["pr_auc_ci"] = ci
    print(f"\n=== {tag} model on the held-out {split.upper()} split ===")
    print(f"  PR-AUC   {metrics['pr_auc']:.4f}  95% CI [{ci['ci_low']:.4f}, "
          f"{ci['ci_high']:.4f}]   (random baseline {y.mean():.5f})")
    print(f"  ROC-AUC  {metrics['roc_auc']:.4f}")
    print(f"  @0.50  precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
          f"f1={metrics['f1']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  confusion @0.50  tn={cm['tn']:,} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")

    # ---- threshold selection: on VALIDATION, never on the split we report ----
    # Picking the argmin of the cost curve on the test split and then quoting
    # the cost at that argmin is selection bias: the threshold has been fitted
    # to the same 71 frauds it is scored against. Measured on this data it
    # understates the true cost by ~11%. So the operating threshold is chosen
    # on validation and merely APPLIED to test.
    X_va, y_va_s = xy("val")
    p_va = predict(booster, X_va)
    sweep_va = sweep_thresholds(y_va_s.to_numpy(), p_va, X_va["Amount"].to_numpy())
    t_selected = float(sweep_va.loc[sweep_va.total_cost.idxmin(), "threshold"])

    sweep = sweep_thresholds(y, p, amounts)
    best = cost_at(y, p, amounts, t_selected)          # honest: val-selected
    oracle = sweep.loc[sweep.total_cost.idxmin()].to_dict()
    oracle["threshold"] = float(oracle["threshold"])    # optimistic: test argmin
    nothing = do_nothing_cost(y, amounts)

    print(f"\n  --- cost model (FP={FP_COST:.0f}/block, FN=actual transaction amount) ---")
    print(f"  do nothing (block none)      : {nothing:>12,.0f}")
    print(f"  at threshold 0.50            : "
          f"{cost_at(y, p, amounts, 0.5)['total_cost']:>12,.0f}")
    print(f"  threshold {best['threshold']:.2f} (SELECTED ON VAL) : "
          f"{best['total_cost']:>12,.0f}"
          f"  (saves {nothing - best['total_cost']:,.0f}, "
          f"{100 * (1 - best['total_cost'] / nothing):.1f}%)")
    print(f"  at that threshold: precision={best['precision']:.4f} "
          f"recall={best['recall']:.4f} fp={int(best['fp'])} fn={int(best['fn'])}")
    bias = best["total_cost"] - oracle["total_cost"]
    print(f"  [reference] test-argmin threshold {oracle['threshold']:.2f} would give "
          f"{oracle['total_cost']:,.0f}, but selecting it on the same split we")
    print(f"              report is selection bias worth {bias:,.0f} "
          f"({100 * bias / max(best['total_cost'], 1):.1f}%). Not used.")

    review_lo = _review_threshold(best["threshold"])
    metrics.update(
        {
            "model": tag,
            "split": split,
            "n_rows": int(len(y)),
            "n_fraud": int(y.sum()),
            "cost_model": {
                "fp_cost_per_block": FP_COST,
                "fn_cost_mode": FN_COST_MODE,
                "chargeback_fee": CHARGEBACK_FEE,
            },
            "cost_optimal": best,
            "threshold_selected_on": "val",
            "cost_oracle_test_argmin": oracle,
            "selection_bias": best["total_cost"] - oracle["total_cost"],
            "cost_at_0_5": cost_at(y, p, amounts, 0.5),
            "do_nothing_cost": nothing,
            "review_band": [review_lo, best["threshold"]],
        }
    )

    (REPORTS / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    sweep.to_csv(REPORTS / f"{tag}_threshold_sweep.csv", index=False)
    plot_pr_curve(y, p, REPORTS / f"pr_curve_{tag}.png",
                  f"Precision-Recall - {tag} model, held-out test")
    plot_cost_curve(sweep, best, nothing, REPORTS / f"cost_curve_{tag}.png")

    if tag == "tuned":
        # Canonical filenames the Streamlit app and README point at.
        for src_n, dst_n in [(f"pr_curve_{tag}.png", "pr_curve.png"),
                             (f"cost_curve_{tag}.png", "cost_curve.png")]:
            (REPORTS / dst_n).write_bytes((REPORTS / src_n).read_bytes())
        SERVING_CONFIG.write_text(json.dumps({
            # Stored relative to models/ so a fresh clone on another machine
            # resolves it. An absolute dev-box path is the classic "works only
            # where it was built" failure.
            "model_path": pathlib.Path(model_path).name,
            "block_threshold": best["threshold"],
            "review_threshold": review_lo,
            "fp_cost": FP_COST,
            "expected_precision_at_block": best["precision"],
            "expected_recall_at_block": best["recall"],
        }, indent=2), encoding="utf-8")
        print(f"[eval] wrote {SERVING_CONFIG} (thresholds the API will serve with)")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tuned", choices=["baseline", "tuned"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    evaluate_model(BASELINE_MODEL if a.model == "baseline" else TUNED_MODEL, a.model, a.split)
