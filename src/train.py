"""Train all three model families on the full train split, then blend them.

Protocol, fixed before any test number is looked at:

    1. Each family is refit on 100% of train using the params Optuna found
       (src/tune.py searched on every fraud plus 10% of the negatives, and
       scored each trial on full validation).
    2. Validation is split by time. The earlier 60% does the fitting work --
       early stopping, blend weights, isotonic calibration. The later 40% is
       held back so src/assess.py can choose the operating threshold on data
       that none of those three fits has touched.
    3. The test split is not read by this file at all. It is opened once, by
       src/assess.py, after everything here is frozen.

Nothing here assumes the ensemble wins. Blend weights are chosen by search, not
set to equal thirds, and whether the blend genuinely beats the best single
model is tested on the held-out split by src/assess.py with a paired bootstrap.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.calibration import Calibration, expected_calibration_error
from src.config import (
    MODELS,
    N_THREADS,
    RESERVE_RAM_GB,
    REPORTS,
    VAL_PRED_DIR,
    available_ram_gb,
)
from src.data import learn_categories, load_split, xy
from src.models import (
    MODEL_NAMES,
    available,
    fit,
    fit_ensemble_weights,
    predict,
    save,
    save_ensemble,
)
from src.fmt import pct, signed_pct
from src.tune import HALF_LIFE_KEY
from src.weights import combined_weights, effective_sample_size
from src.config import RANDOM_SEED  # noqa: E402
from src.schema import FEATURES, LABEL

# Fraction of validation used for FITTING (early stopping, blend weights,
# calibration). The remainder is held back for threshold selection.
VAL_FIT_FRACTION = 0.60


def _thin(X, y, dates, keep_frac: float):
    """Keep every fraud, thin the legitimate rows.

    The positives are what PR-AUC is made of, so they are never touched. Only
    negatives are dropped, and both the class and recency weights are then
    recomputed from whatever sample the model actually sees.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg = rng.choice(neg, size=int(len(neg) * keep_frac), replace=False)
    sel = np.sort(np.concatenate([pos, neg]))
    d = dates.iloc[sel] if dates is not None else None
    return X.iloc[sel], y[sel], d, len(pos)


def fit_within_memory(name, X, y, X_val, y_val, params, dates=None,
                      half_life=0):
    """Fit on ALL the data, and fall back only if the machine actually refuses.

    Pre-emptively subsampling because memory *might* run out throws away rows
    that would usually have fitted fine. So we try the full split first and
    thin only after a real failure -- losing some negatives beats losing the
    run, but not when it was never necessary.

    Any thinning is reported loudly and recorded in the training summary: a
    silently subsampled model is a lie in the metrics table.
    """
    attempts = [(1.0, "full")] + [(0.5, "50% of negatives"),
                                  (0.25, "25% of negatives")]
    last_exc = None
    for frac, describe in attempts:
        if frac == 1.0:
            Xf, yf, df, n_pos, rows = X, y, dates, int(y.sum()), len(y)
        else:
            Xf, yf, df, n_pos = _thin(X, y, dates, frac)
            rows = len(yf)
            print(f"[train] retrying {name} on {describe}: {rows:,} rows, "
                  f"all {n_pos:,} frauds kept", flush=True)
        w = combined_weights(yf, df, half_life)
        try:
            model = fit(name, Xf, yf, X_val, y_val, params=params,
                        sample_weight=w)
            return model, {"rows_used": rows, "frauds_used": n_pos,
                           "subsampled": frac < 1.0, "negative_fraction": frac,
                           "recency_half_life_days": half_life,
                           "effective_sample_size": round(
                               effective_sample_size(w), 1)}
        except Exception as exc:  # noqa: BLE001 - MemoryError and library OOMs
            msg = str(exc)[:120]
            if not _is_memory_error(exc):
                raise
            last_exc = exc
            print(f"[train] {name} ran out of memory on {describe} "
                  f"({type(exc).__name__}: {msg})", flush=True)
            del Xf, yf
            gc.collect()
    raise RuntimeError(f"{name} could not be trained within memory") from last_exc


def _is_memory_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in
               ("memory", "bad_alloc", "cannot allocate", "out of ram"))


def load_best_params(name: str) -> dict | None:
    path = MODELS / f"best_params_{name}.json"
    if not path.exists():
        print(f"[train] {name}: no tuned params found, using defaults "
              f"(run `python run.py tune` first)")
        return None, 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"[train] {name}: using tuned params "
          f"(search val PR-AUC {pct(payload.get('best_val_pr_auc'))})")
    params = dict(payload.get("best_params") or {})
    # The recency half-life was tuned alongside the model parameters but is
    # consumed by the weighting code, not by the booster -- passing it through
    # would be an unknown-parameter error.
    half_life = params.pop(HALF_LIFE_KEY, 0)
    if half_life:
        print(f"[train] {name}: recency half-life {half_life} days "
              f"(a fraud that old counts half as much as one today)")
    else:
        print(f"[train] {name}: no recency decay selected -- all history "
              f"weighted equally")
    return params, half_life


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODEL_NAMES))
    ap.add_argument("--force", action="store_true",
                    help="refit even if a checkpoint exists")
    args = ap.parse_args()

    train = load_split("train", columns=FEATURES + [LABEL, "date"])
    cats = learn_categories(train)
    X_tr, y_tr = train[FEATURES], train[LABEL].to_numpy()
    dates_tr = train["date"]
    print(f"[train] train {X_tr.shape}, {int(y_tr.sum()):,} frauds "
          f"({y_tr.mean():.4%})")
    del train

    X_val, y_val = xy("val", cats)
    y_val = y_val.to_numpy()

    # Validation is split BY TIME into two slices with different jobs:
    #   val_fit  (earlier 60%) -- early stopping, blend weights, calibration
    #   val_sel  (later   40%) -- reserved for threshold selection in assess.py
    # Using one slice for both would let the operating threshold be tuned to
    # the calibrator's own fitting noise. val.parquet is already time-ordered.
    cut = int(len(y_val) * VAL_FIT_FRACTION)
    X_fit, y_fit = X_val.iloc[:cut], y_val[:cut]
    X_sel, y_sel = X_val.iloc[cut:], y_val[cut:]
    print(f"[train] val   {X_val.shape}, {int(y_val.sum()):,} frauds")
    print(f"[train]   val_fit {len(y_fit):,} rows ({int(y_fit.sum())} frauds) "
          f"-- early stopping, blend weights, calibration")
    print(f"[train]   val_sel {len(y_sel):,} rows ({int(y_sel.sum())} frauds) "
          f"-- held back for threshold selection")
    (MODELS / "val_split.json").write_text(json.dumps(
        {"val_fit_fraction": VAL_FIT_FRACTION, "cut_index": cut,
         "val_fit_rows": int(cut), "val_sel_rows": int(len(y_val) - cut)}),
        encoding="utf-8")

    val_preds: dict[str, np.ndarray] = {}
    rows = []
    for name in args.models:
        t0 = time.time()
        pred_path = VAL_PRED_DIR / f"{name}.npy"

        # CHECKPOINT. Each model is saved the moment it is fitted, along with
        # its validation predictions, so an interrupted run resumes at the next
        # model instead of refitting hours of work.
        if not args.force and name in available() and pred_path.exists():
            p = np.load(pred_path)
            if len(p) == len(y_val):
                val_preds[name] = p
                pr = float(average_precision_score(y_val, p))
                rows.append({"model": name, "val_pr_auc": pr,
                             "val_roc_auc": float(roc_auc_score(y_val, p)),
                             "train_seconds": 0.0, "restored": True})
                print(f"[train] {name:<10} restored from checkpoint, "
                      f"val PR-AUC {pct(pr)}")
                continue

        free = available_ram_gb()
        print(f"[train] fitting {name} "
              f"({free:.1f}gb RAM free)" if free else f"[train] fitting {name}",
              flush=True)
        params, half_life = load_best_params(name)
        model, fit_info = fit_within_memory(name, X_tr, y_tr, X_fit, y_fit,
                                            params, dates=dates_tr,
                                            half_life=half_life)
        save(name, model)
        p = predict(name, model, X_val)
        np.save(pred_path, p)
        del model
        gc.collect()

        val_preds[name] = p
        pr = float(average_precision_score(y_val, p))
        roc = float(roc_auc_score(y_val, p))
        secs = time.time() - t0
        rows.append({"model": name, "val_pr_auc": pr, "val_roc_auc": roc,
                     "train_seconds": round(secs, 1), **fit_info})
        note = ("  [SUBSAMPLED: " + f"{fit_info['rows_used']:,} rows]"
                if fit_info["subsampled"] else "")
        print(f"[train] {name:<10} val PR-AUC {pct(pr)}  ROC-AUC {pct(roc)}  "
              f"({secs:.0f}s){note}")

    # Blend weights are chosen on val_fit only, never on the slice that will
    # choose the threshold.
    fit_preds = {n: p[:cut] for n, p in val_preds.items()}
    weights = fit_ensemble_weights(fit_preds, y_fit)
    from src.models import ensemble_predict

    ens = ensemble_predict(val_preds, weights)
    ens_pr = float(average_precision_score(y_val, ens))

    # Calibrate the blended score on val_fit, and measure the effect on the
    # held-back slice so the number is not self-reported.
    cal = Calibration.fit(ens[:cut], y_fit)
    before = expected_calibration_error(ens[cut:], y_sel)
    after = expected_calibration_error(cal.predict(ens[cut:]), y_sel)
    cal.save()
    pr_after = float(average_precision_score(y_sel, cal.predict(ens[cut:])))
    pr_before = float(average_precision_score(y_sel, ens[cut:]))
    print(f"[train] calibration (isotonic, fitted on val_fit, measured on val_sel)")
    print(f"    expected calibration error {pct(before, 4)} -> {pct(after, 4)}")
    print(f"    PR-AUC {pct(pr_before)} -> {pct(pr_after)} "
          f"(monotone map: ranking is unchanged by construction)")
    rows.append({"model": "ensemble", "val_pr_auc": ens_pr,
                 "val_roc_auc": float(roc_auc_score(y_val, ens)),
                 "train_seconds": 0.0})

    best_single = max(r["val_pr_auc"] for r in rows if r["model"] != "ensemble")
    save_ensemble(weights, {
        "val_pr_auc": ens_pr,
        "best_single_val_pr_auc": best_single,
        "beats_best_single_on_val": bool(ens_pr > best_single),
        "calibration": {"method": cal.method, "ece_before": before,
                        "ece_after": after},
        "members": list(val_preds),
        "note": "Weights chosen on validation. Whether the blend is genuinely "
                "better than the best single model is tested on the held-out "
                "split by src/evaluate.py, with a paired bootstrap.",
    })

    baseline = float(y_val.mean())
    print("\n[train] validation summary (random baseline PR-AUC "
          f"= {baseline:.5f})")
    for r in sorted(rows, key=lambda r: -r["val_pr_auc"]):
        print(f"    {r['model']:<10} PR-AUC {r['val_pr_auc']:.4f}  "
              f"ROC-AUC {r['val_roc_auc']:.4f}")
    if ens_pr > best_single:
        print(f"[train] ensemble improves on the best single model on val "
              f"(+{ens_pr - best_single:.4f}) -- to be confirmed on test")
    else:
        print(f"[train] ensemble does NOT improve on the best single model on "
              f"val ({ens_pr - best_single:+.4f})")

    (REPORTS / "training_summary.json").write_text(json.dumps(
        {"models": rows, "ensemble_weights": weights,
         "val_random_baseline": baseline}, indent=2), encoding="utf-8")
    print("[train] wrote reports/training_summary.json")


if __name__ == "__main__":
    main()
