"""Open the held-out test split once, and report whatever it says.

Everything upstream -- features, hyper-parameters, blend weights, calibration,
the operating threshold -- is fixed using train and validation only. This is the
single place the test split is read.

The order of operations matters, and it is the order production uses:

    1. Choose WHICH model to ship, on val_fit.
    2. Fit the calibrator for that model, on val_fit.
    3. Choose the operating threshold on CALIBRATED scores over val_sel -- a
       slice no model, no blend weight and no calibrator has been fitted on.
    4. Apply all of it, unchanged, to test.

Step 3 is easy to get subtly wrong. Selecting a threshold on uncalibrated
scores and then serving calibrated ones means the number the API compares
against is not the number the threshold was chosen for; the operating point
silently moves. Everything below is calibrated before any threshold is touched.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score

from src.calibration import Calibration, expected_calibration_error
from src.config import (
    CHARGEBACK_FEE,
    FN_COST_MODE,
    FP_COST,
    MODELS,
    REPORTS,
    REVIEW_CAPACITY_RATE,
    SERVING_CONFIG,
)
from src.data import load_categories, load_split, xy
from src.fmt import money, pct, signed_pct
from src.metrics import (
    bootstrap_pr_auc,
    core_metrics,
    cost_at,
    do_nothing_cost,
    paired_bootstrap_delta,
    plot_cost_curve,
    plot_pr_curve,
    sweep_thresholds,
)
from src.models import available, ensemble_predict, load, load_ensemble, predict

MODEL_VERSION = "frm-1.0.0"


def _amounts(split: str) -> np.ndarray:
    return load_split(split, columns=["abs_amount"])["abs_amount"].to_numpy("float64")


def _predict_all(split: str, cats, weights: dict) -> tuple[dict, np.ndarray]:
    X, y = xy(split, cats)
    preds = {n: predict(n, load(n), X) for n in available()}
    members = {k: v for k, v in preds.items() if weights.get(k, 0) > 0}
    if members:
        preds["ensemble"] = ensemble_predict(members, weights)
    return preds, y.to_numpy()


def _val_cut(n: int) -> int:
    path = MODELS / "val_split.json"
    if path.exists():
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["cut_index"])
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return int(n * 0.6)


def _review_threshold(scores: np.ndarray, block_threshold: float) -> float:
    """Lower edge of the review band, sized by reviewer CAPACITY.

    A merchant's review team can look at a fixed share of traffic per day. That
    operational limit -- not a magic constant -- decides how wide the band is.
    """
    t = float(np.quantile(scores, 1.0 - REVIEW_CAPACITY_RATE))
    t = max(min(t, block_threshold * 0.99), 0.0)
    share = float((scores >= t).mean())
    print(f"[assess] review band sized by capacity: score >= {pct(t)} sends "
          f"{pct(share)} of traffic to a human "
          f"(target {pct(REVIEW_CAPACITY_RATE)})")
    return round(t, 6)


def main() -> None:
    cats = load_categories()
    weights = load_ensemble().get("weights", {})

    print("[assess] scoring validation")
    p_val, y_val = _predict_all("val", cats, weights)
    a_val = _amounts("val")
    cut = _val_cut(len(y_val))
    y_fit, y_sel = y_val[:cut], y_val[cut:]
    a_sel = a_val[cut:]
    print(f"[assess]   val_fit {len(y_fit):,} rows ({int(y_fit.sum())} frauds) "
          f"-- model choice + calibration")
    print(f"[assess]   val_sel {len(y_sel):,} rows ({int(y_sel.sum())} frauds) "
          f"-- threshold selection")

    # ---- 1. which model ships, decided on val_fit ----
    fit_scores = {n: float(average_precision_score(y_fit, p[:cut]))
                  for n, p in p_val.items()}
    shipped = max(fit_scores, key=fit_scores.get)
    print("\n[assess] PR-AUC on val_fit (this is what selects the model)")
    for n, s in sorted(fit_scores.items(), key=lambda kv: -kv[1]):
        print(f"    {n:<10} {pct(s):>8}" + ("  <-- shipped" if n == shipped else ""))

    # ---- 2. calibrate the shipped scorer on val_fit ----
    cal = Calibration.fit(p_val[shipped][:cut], y_fit)
    ece_before = expected_calibration_error(p_val[shipped][cut:], y_sel)
    ece_after = expected_calibration_error(cal.predict(p_val[shipped][cut:]), y_sel)
    cal.save()
    print(f"\n[assess] isotonic calibration for '{shipped}' (fitted on val_fit, "
          f"measured on val_sel)")
    print(f"    expected calibration error {pct(ece_before, 4)} -> {pct(ece_after, 4)}")

    # ---- 3. threshold on CALIBRATED val_sel ----
    sel_cal = cal.predict(p_val[shipped][cut:])
    sweep_sel = sweep_thresholds(y_sel, sel_cal, a_sel)
    t_block = float(sweep_sel.loc[sweep_sel.total_cost.idxmin(), "threshold"])
    review_lo = _review_threshold(sel_cal, t_block)

    # ---- 4. one look at test, everything applied unchanged ----
    print("\n[assess] scoring the held-out TEST split (first and only look)")
    p_test, y_test = _predict_all("test", cats, weights)
    a_test = _amounts("test")
    p_test_cal = {n: cal.predict(p) if n == shipped else p for n, p in p_test.items()}

    # The candidate comparison uses UNCALIBRATED scores for every model, so the
    # table compares like with like. Calibrating only the shipped model would
    # quietly handicap it: isotonic regression is monotone, so it cannot
    # reorder anything, but its flat regions merge distinct scores into ties
    # (measured here: 101,690 distinct values collapse to 8,328), and ties cost
    # ~0.15pp of PR-AUC. That is a fair price for an expected calibration error
    # of 0.645% -> 0.0117%, but it is not a reason to report the shipped model
    # as worse than its rivals.
    results = {}
    for name, p in p_test.items():
        m = core_metrics(y_test, p, 0.5)
        # 500 resamples, not 2000. The interval's WIDTH is set by the number
        # of positives (1,409 frauds), not by the resample count; 500 is ample
        # to place the 2.5/97.5 quantiles and costs a quarter of the compute on
        # an 800k-row split.
        m["pr_auc_ci"] = bootstrap_pr_auc(y_test, p, n_boot=500)
        # ...but the operating-point numbers use the calibrated score for the
        # shipped model, because that is what the service actually thresholds.
        scored = p_test_cal.get(name, p) if name == shipped else p
        m["cost_at_selected_threshold"] = cost_at(y_test, scored, a_test, t_block)
        m["calibrated_for_serving"] = bool(name == shipped)
        results[name] = m

    shipped_p = p_test_cal[shipped]
    sweep_test = sweep_thresholds(y_test, shipped_p, a_test)
    best = cost_at(y_test, shipped_p, a_test, t_block)
    oracle = sweep_test.loc[sweep_test.total_cost.idxmin()].to_dict()
    oracle["threshold"] = float(oracle["threshold"])
    nothing = do_nothing_cost(y_test, a_test)
    baseline = float(y_test.mean())

    print(f"\n[assess] held-out test: {len(y_test):,} transactions, "
          f"{int(y_test.sum()):,} frauds ({pct(baseline, 3)})")
    print(f"    PR-AUC {pct(results[shipped]['pr_auc'])}  "
          f"(random baseline {pct(baseline, 3)}, "
          f"lift {results[shipped]['pr_auc'] / max(baseline, 1e-12):,.0f}x)")
    print(f"    ROC-AUC {pct(results[shipped]['roc_auc'])}")

    print(f"\n[assess] cost model: FP={FP_COST:.0f} per wrong block, FN=actual amount")
    print(f"    do nothing                        {money(nothing):>14}")
    print(f"    at threshold {pct(t_block)} (chosen on val_sel) "
          f"{money(best['total_cost']):>14}")
    print(f"    saves {money(nothing - best['total_cost'])} "
          f"({pct(1 - best['total_cost'] / max(nothing, 1), 1)} of avoidable loss)")
    print(f"    precision {pct(best['precision'])}  recall {pct(best['recall'])}  "
          f"fp={int(best['fp'])}  fn={int(best['fn'])}  "
          f"block rate {pct(best['block_rate'], 3)}")
    bias = best["total_cost"] - oracle["total_cost"]
    print(f"    [reference] the test-argmin threshold {pct(oracle['threshold'])} would "
          f"give {money(oracle['total_cost'])}; choosing it on the very split we "
          f"report would be worth {money(bias)} of selection bias. Not used.")

    # ---- is the blend actually better than the best single model? ----
    singles = {n: s for n, s in fit_scores.items() if n != "ensemble"}
    best_single = max(singles, key=singles.get)
    cmp_block = {}
    if "ensemble" in p_test_cal and best_single != "ensemble":
        delta = paired_bootstrap_delta(y_test, p_test[best_single],
                                       p_test["ensemble"], n_boot=500)
        cmp_block = {"best_single": best_single, **delta}
        print(f"\n[assess] ensemble vs best single ({best_single}) on test")
        print(f"    {best_single:<10} PR-AUC {pct(results[best_single]['pr_auc'])}")
        print(f"    ensemble   PR-AUC {pct(results['ensemble']['pr_auc'])}")
        print(f"    paired delta {signed_pct(delta['delta_point'])}  95% CI "
              f"[{signed_pct(delta['ci_low'])}, {signed_pct(delta['ci_high'])}]")
        if not delta["significant_at_95"]:
            print("    -> NOT a statistically significant improvement. "
                  "Reported as such.")

    # ---- 5. the 1% check set: a SECOND, even fresher held-out period ----
    # test ends 2019-09; this is the month after it. Nothing in the pipeline has
    # seen it, and it is reported separately rather than merged into test, so it
    # works as an independent confirmation that the test number was not luck.
    demo_block = {}
    try:
        p_demo, y_demo = _predict_all("demo", cats, weights)
        a_demo = _amounts("demo")
        dp = cal.predict(p_demo[shipped]) if shipped in p_demo else None
        if dp is not None and y_demo.sum() > 0:
            dm = core_metrics(y_demo, dp, 0.5)
            dcost = cost_at(y_demo, dp, a_demo, t_block)
            dbase = float(y_demo.mean())
            demo_block = {"rows": int(len(y_demo)), "frauds": int(y_demo.sum()),
                          "base_rate": dbase, "pr_auc": dm["pr_auc"],
                          "roc_auc": dm["roc_auc"],
                          "cost_at_selected_threshold": dcost}
            print("")
            print("[assess] 1% check set (the month AFTER test, never touched)")
            print(f"    {len(y_demo):,} transactions, {int(y_demo.sum()):,} frauds "
                  f"({pct(dbase, 3)})")
            print(f"    PR-AUC {pct(dm['pr_auc'])}  "
                  f"(test was {pct(results[shipped]['pr_auc'])})")
            print(f"    at the shipped threshold: precision "
                  f"{pct(dcost['precision'])}  recall {pct(dcost['recall'])}")
            gap = dm["pr_auc"] - results[shipped]["pr_auc"]
            if abs(gap) > 0.15:
                print(f"    NOTE: {signed_pct(gap)} away from the test result -- "
                      f"a gap that size means the operating point is not stable "
                      f"across adjacent months.")
    except FileNotFoundError:
        print("[assess] no demo split found -- skipping the 1% check")

    plot_pr_curve(y_test, shipped_p, REPORTS / "pr_curve.png",
                  f"Precision-Recall - {shipped}, held-out test")
    plot_cost_curve(sweep_test, best, nothing, REPORTS / "cost_curve.png")

    payload = {
        "model_version": MODEL_VERSION, "shipped": shipped,
        "selected_on": "val_fit", "threshold_selected_on": "val_sel",
        "val_fit_pr_auc": fit_scores, "test": results,
        "test_random_baseline": baseline,
        "calibration": {"method": cal.method, "ece_before": ece_before,
                        "ece_after": ece_after},
        "cost_model": {"fp_cost_per_block": FP_COST, "fn_cost_mode": FN_COST_MODE,
                       "chargeback_fee": CHARGEBACK_FEE},
        "cost_optimal": best, "cost_oracle_test_argmin": oracle,
        "selection_bias": bias, "do_nothing_cost": nothing,
        "review_band": [review_lo, t_block], "ensemble_weights": weights,
        "ensemble_vs_best_single": cmp_block,
        "test_rows": int(len(y_test)), "test_frauds": int(y_test.sum()),
        "check_set": demo_block,
    }
    (REPORTS / "assessment.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    sweep_test.to_csv(REPORTS / "threshold_sweep.csv", index=False)

    SERVING_CONFIG.write_text(json.dumps({
        "shipped_model": shipped, "ensemble_weights": weights,
        "block_threshold": t_block, "review_threshold": review_lo,
        "fp_cost": FP_COST, "model_version": MODEL_VERSION,
        "expected_precision_at_block": best["precision"],
        "expected_recall_at_block": best["recall"],
    }, indent=2), encoding="utf-8")
    print(f"\n[assess] wrote reports/assessment.json and {SERVING_CONFIG.name}")
    print(f"[assess] serving: {shipped}  block at {pct(t_block)}  "
          f"review from {pct(review_lo)}")


if __name__ == "__main__":
    main()
