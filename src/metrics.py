"""Metrics, cost modelling and plots.

These are the statistical core of the project and they are deliberately
schema-agnostic: every function takes plain (y, p, amounts) arrays and knows
nothing about which dataset or which model produced them. That is why they
survived the move from the anonymised-PCA dataset to this one unchanged.

The cost model is the part worth defending out loud:
  * a missed fraud costs the merchant the FULL amount of that transaction, not
    an average -- the amount distribution is skewed and an average flatters us;
  * a wrongly blocked legitimate payment costs a flat FP_COST in support and
    lost goodwill. It is a stated assumption, not a fitted parameter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import CHARGEBACK_FEE, FN_COST_MODE, FP_COST


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


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
