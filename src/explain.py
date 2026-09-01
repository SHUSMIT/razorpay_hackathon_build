"""SHAP explanations.

Implementation note worth stating out loud: this uses XGBoost's built-in
TreeSHAP (`Booster.predict(..., pred_contribs=True)`), not the `shap` Python
package. It is the same TreeSHAP algorithm -- Lundberg's implementation was
upstreamed into XGBoost's C++ core -- but it has no numba/llvmlite dependency,
so the service starts on a locked-down machine and per-request explanation
stays fast enough to sit in the request path (single-digit ms).

`shap` is still used, if it imports, for the prettier global summary beeswarm.
Values are exact log-odds contributions either way: contributions + bias sums
to the raw margin, which `verify_additivity` asserts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import REPORTS, TUNED_MODEL
from src.train import xy

# Human-readable labels. V1..V28 are anonymised PCA components published by the
# dataset authors -- we deliberately do NOT invent business meanings for them.
FRIENDLY = {
    "Amount": "transaction amount",
    "log_amount": "transaction amount (log scale)",
    "hour_of_day": "hour of day",
    "hour_sin": "time of day (cyclical)",
    "hour_cos": "time of day (cyclical)",
}


def friendly(name: str) -> str:
    if name in FRIENDLY:
        return FRIENDLY[name]
    if name.startswith("V"):
        return f"anonymised behaviour component {name}"
    return name


def shap_contribs(booster: xgb.Booster, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP values. Returns (contribs [n, n_features], bias)."""
    d = xgb.DMatrix(X, feature_names=list(X.columns))
    raw = booster.predict(d, pred_contribs=True)
    return raw[:, :-1], float(raw[0, -1])


def top_factors(contribs_row: np.ndarray, columns: list[str], k: int = 3) -> list[dict]:
    """Top-k features pushing this single prediction, by absolute contribution."""
    order = np.argsort(np.abs(contribs_row))[::-1][:k]
    out = []
    for i in order:
        v = float(contribs_row[i])
        out.append({
            "feature": columns[i],
            "label": friendly(columns[i]),
            "shap_value": round(v, 6),
            "direction": "increases risk" if v > 0 else "decreases risk",
        })
    return out


def explain_one(booster: xgb.Booster, X_row: pd.DataFrame, k: int = 3) -> list[dict]:
    contribs, _ = shap_contribs(booster, X_row)
    return top_factors(contribs[0], list(X_row.columns), k)


def verify_additivity(booster: xgb.Booster, X: pd.DataFrame, tol: float = 1e-3) -> float:
    """SHAP values must sum to the raw margin. Cheap, and catches wiring bugs."""
    d = xgb.DMatrix(X, feature_names=list(X.columns))
    raw = booster.predict(d, pred_contribs=True)
    margin = booster.predict(d, output_margin=True)
    err = float(np.max(np.abs(raw.sum(axis=1) - margin)))
    assert err < tol, f"SHAP additivity violated: max error {err}"
    return err


# ------------------------------------------------------------------- plotting
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def global_summary(n_sample: int = 2000, seed: int = 42) -> dict:
    booster = xgb.Booster()
    booster.load_model(TUNED_MODEL)

    X, y = xy("test")
    # Over-sample the fraud rows: explaining 2000 random transactions on a
    # 0.17%-positive dataset would show us almost nothing about fraud.
    frauds = X[y == 1]
    legit = X[y == 0].sample(min(n_sample - len(frauds), (y == 0).sum()), random_state=seed)
    sample = pd.concat([frauds, legit])
    print(f"[explain] sample = {len(sample)} rows ({len(frauds)} fraud, {len(legit)} legit)")

    err = verify_additivity(booster, sample.head(200))
    print(f"[explain] TreeSHAP additivity check passed (max error {err:.2e})")

    contribs, bias = shap_contribs(booster, sample)
    cols = list(sample.columns)
    mean_abs = np.abs(contribs).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    ranking = [
        {"feature": cols[i], "label": friendly(cols[i]),
         "mean_abs_shap": float(mean_abs[i]),
         "mean_shap": float(contribs[:, i].mean())}
        for i in order
    ]
    print("\n[explain] top 10 global drivers (mean |SHAP|)")
    for r in ranking[:10]:
        print(f"  {r['feature']:<12} {r['mean_abs_shap']:.4f}   {r['label']}")

    _plot_summary(contribs, sample, cols, order[:15])
    _plot_beeswarm_if_available(contribs, sample, cols)

    payload = {"model_version_file": str(TUNED_MODEL), "bias_log_odds": bias,
               "n_sample": int(len(sample)), "additivity_max_error": err,
               "ranking": ranking}
    (REPORTS / "shap_global.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[explain] wrote {REPORTS / 'shap_global.json'}")
    return payload


def _plot_summary(contribs, sample, cols, top_idx):
    plt = _mpl()
    names = [cols[i] for i in top_idx][::-1]
    vals = [np.abs(contribs[:, i]).mean() for i in top_idx][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names, vals, color="#2563eb")
    ax.set_xlabel("mean |SHAP value|  (average impact on the risk log-odds)")
    ax.set_title("Global feature importance - TreeSHAP, tuned model, test sample")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "shap_summary.png", dpi=140)
    plt.close(fig)
    print(f"[explain] wrote {REPORTS / 'shap_summary.png'}")


def _plot_beeswarm_if_available(contribs, sample, cols):
    """Nicer beeswarm when `shap` imports; skipped silently when it does not."""
    try:
        import shap  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"[explain] `shap` package unavailable ({type(exc).__name__}); "
              "bar summary only. Values above are still exact TreeSHAP.")
        return
    plt = _mpl()
    shap.summary_plot(contribs, sample, feature_names=cols, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_beeswarm.png", dpi=140)
    plt.close("all")
    print(f"[explain] wrote {REPORTS / 'shap_beeswarm.png'}")


if __name__ == "__main__":
    global_summary()
