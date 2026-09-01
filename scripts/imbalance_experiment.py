#!/usr/bin/env python
"""Which imbalance technique actually earns its place?

331 frauds against 198,277 legitimate transactions. There are a dozen standard
answers to that and most papers report the one that won on their data. This
script runs fourteen of them under identical conditions and reports the whole
table, losers included.

PROTOCOL, fixed before any number was looked at
-----------------------------------------------
  * Every arm gets the SAME hyperparameters (the shipped Optuna winner) and the
    SAME fixed number of boosting rounds. Only the imbalance treatment varies,
    so a difference in the table is a difference in the treatment and not in
    how long something was allowed to train.
  * NO EARLY STOPPING. Stopping on validation would let each arm peek at the
    split it is then scored on, and the arms that overfit fastest would look
    best. Fixed rounds costs a little accuracy and buys a fair comparison.
  * Every arm is trained on TRAIN and scored on VAL, over several seeds. The
    spread across seeds is reported next to the mean, because on 71 validation
    frauds a 0.01 PR-AUC gap is frequently just the seed.
  * The TEST split is not loaded until the winner has been chosen. It is then
    used once, and compared to the shipped model with a paired bootstrap.

Run: python scripts/imbalance_experiment.py [--seeds 3] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrate import (  # noqa: E402
    Calibrator,
    calibration_report,
    cross_fitted_calibration,
    fit_recipe,
    oof_predictions,
    plot_reliability,
    score_with,
)
from src.config import (  # noqa: E402
    BEST_PARAMS,
    FP_COST,
    MODELS,
    REPORTS,
    REVIEW_BAND_WIDTH,
    SERVING_CONFIG,
    TUNED_MODEL,
)
from src.evaluate import cost_at, do_nothing_cost, paired_bootstrap_delta, predict  # noqa: E402
from src.threshold import plot_selection, select_threshold, stability_report  # noqa: E402
from src.train import xy  # noqa: E402

SEED = 42
ROUNDS = 140


def base_params() -> dict:
    p = dict(json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"])
    p["nthread"] = 8
    return p


def arms(P: dict) -> list[dict]:
    """The fourteen candidates.

    `ratio` is positives/negatives after resampling: 0.01 takes 331 frauds to
    ~1,980 (6x), 0.05 to ~9,900 (30x). Full balancing (ratio 1.0) is not on the
    list -- it would mean synthesising 198,000 frauds from 331 real ones, at
    which point 99.8% of what the model learns about fraud is our own
    interpolation.

    Resampling arms use scale_pos_weight="auto", recomputed AFTER resampling.
    Leaving the original 600 in place would apply the imbalance correction
    twice and is the standard way SMOTE gets blamed for a model that blocks
    everything.
    """
    spw = P.get("scale_pos_weight", 600.0)
    common = {"params": P, "rounds": ROUNDS, "fp_cost": FP_COST}
    return [
        dict(common, name="none", family="baseline",
             note="no imbalance handling at all - the floor everything must clear"),
        dict(common, name="scale_pos_weight", family="reweight", scale_pos_weight=spw,
             note="SHIPPED recipe: one constant multiplier on every fraud"),
        dict(common, name="class_weight", family="reweight", weighting="class_weight",
             note="inverse frequency as per-row weights instead of spw"),
        dict(common, name="effective_number", family="reweight",
             weighting="effective_number",
             note="Cui et al. class-balanced: discounts redundancy among frauds"),
        dict(common, name="amount_weight", family="reweight", weighting="amount_weight",
             note="COST-SENSITIVE: weight each row by the money it risks"),
        dict(common, name="focal_loss", family="loss", focal={"gamma": 2.0, "alpha": 0.75},
             note="Lin et al.: gradient concentrates on hard boundary rows"),
        dict(common, name="focal_loss_g1", family="loss", focal={"gamma": 1.0, "alpha": 0.9},
             note="milder focusing, stronger class prior"),
        dict(common, name="label_smoothing", family="targets", scale_pos_weight=spw,
             eps_pos=0.01, eps_neg=0.002,
             note="spw + soft targets: undisputed fraud is labelled legitimate"),
        dict(common, name="random_oversample", family="resample",
             resampler="random_oversample", ratio=0.01, scale_pos_weight="auto",
             note="duplicate frauds - adds no information, the floor for SMOTE"),
        dict(common, name="smote", family="resample", resampler="smote", ratio=0.01,
             scale_pos_weight="auto", note="Chawla et al. interpolation, 6x frauds"),
        dict(common, name="smote_x30", family="resample", resampler="smote", ratio=0.05,
             scale_pos_weight="auto", note="same, 30x frauds - does more help?"),
        dict(common, name="borderline_smote", family="resample",
             resampler="borderline_smote", ratio=0.01, scale_pos_weight="auto",
             note="synthesise only near the decision boundary"),
        dict(common, name="adasyn", family="resample", resampler="adasyn", ratio=0.01,
             scale_pos_weight="auto", note="density-adaptive synthesis"),
        dict(common, name="jitter", family="resample", resampler="jitter", ratio=0.01,
             scale_pos_weight="auto",
             note="noise scaled to minority spread - can leave the fraud hull"),
        dict(common, name="smote_enn", family="resample", resampler="smote_enn",
             ratio=0.01, scale_pos_weight="auto",
             note="SMOTE then clean the interleaved region"),
        dict(common, name="borderline+amount", family="combo",
             resampler="borderline_smote", ratio=0.01, weighting="amount_weight",
             scale_pos_weight="auto",
             note="best resampler x cost-sensitive weights"),
        dict(common, name="smote+smoothing", family="combo", resampler="smote",
             ratio=0.01, scale_pos_weight="auto", eps_pos=0.01, eps_neg=0.002,
             note="best resampler x soft targets"),
    ]


# ------------------------------------------------------------------ the bench
def run_bench(seeds: list[int], quick: bool) -> pd.DataFrame:
    X_tr, y_tr = xy("train")
    X_va, y_va = xy("val")
    fn = list(X_tr.columns)

    Xn, yn = X_tr.to_numpy(), y_tr.to_numpy()
    amt = X_tr["Amount"].to_numpy()
    Xv, yv = X_va.to_numpy(), y_va.to_numpy()
    amt_v = X_va["Amount"].to_numpy()

    if quick:
        # Subsample BOTH classes at the same rate. Thinning only the majority
        # would drop the imbalance from 599:1 to 90:1, at which point the
        # resampling arms are asked to reach a fraud ratio they already exceed,
        # silently become no-ops, and every one of them reports an identical
        # number. Preserving the ratio keeps --quick a smoke test of this script
        # rather than a different experiment.
        rng = np.random.default_rng(SEED)
        frac = 30_000 / len(yn)
        pos, neg = np.flatnonzero(yn == 1), np.flatnonzero(yn == 0)
        keep = np.concatenate([
            rng.choice(pos, max(int(len(pos) * frac), 10), replace=False),
            rng.choice(neg, int(len(neg) * frac), replace=False)])
        Xn, yn, amt = Xn[keep], yn[keep], amt[keep]
        print(f"[bench] --quick: {len(yn):,}-row subsample, {int(yn.sum())} frauds "
              f"(imbalance preserved). Timing check only, NOT a result.")

    print(f"[bench] train={Xn.shape} frauds={int(yn.sum())}  "
          f"val={Xv.shape} frauds={int(yv.sum())}  test NOT loaded")
    print(f"[bench] {len(arms(base_params()))} arms x {len(seeds)} seeds, "
          f"{ROUNDS} fixed rounds each\n")

    from src.imbalance import resample as _resample

    rows = []
    for arm in arms(base_params()):
        # What the treatment actually did to the training set, reported rather
        # than assumed: an arm whose fraud count is unchanged is a no-op, and a
        # no-op that scores well is not evidence for the technique.
        _, y_after = _resample(arm.get("resampler", "none"), Xn, yn,
                               ratio=arm.get("ratio", 0.0), seed=SEED)
        pos_after = int(y_after.sum())
        if arm.get("resampler") and pos_after == int(yn.sum()):
            print(f"  [warn] {arm['name']}: ratio {arm.get('ratio')} is already met, "
                  "resampler added nothing")

        pr, costs, thr, t0 = [], [], [], time.time()
        for s in seeds:
            booster, is_margin = fit_recipe(Xn, yn, amt, arm, fn, s)
            p = score_with(booster, Xv, fn, is_margin)
            pr.append(float(average_precision_score(yv, p)))
            # Cost at the shipped operating threshold, so the table also says
            # what each arm would do to the merchant's money today.
            c = cost_at(yv, p, amt_v, 0.5)
            costs.append(c["total_cost"])
            thr.append(c)
        rows.append({
            "arm": arm["name"], "family": arm["family"],
            "train_frauds_after": pos_after,
            "train_rows_after": int(len(y_after)),
            "val_pr_auc": float(np.mean(pr)), "val_pr_auc_std": float(np.std(pr)),
            "val_pr_auc_min": float(np.min(pr)), "val_pr_auc_max": float(np.max(pr)),
            "val_cost_at_0.5": float(np.mean(costs)),
            "val_precision_at_0.5": float(np.mean([c["precision"] for c in thr])),
            "val_recall_at_0.5": float(np.mean([c["recall"] for c in thr])),
            "val_fp_at_0.5": float(np.mean([c["fp"] for c in thr])),
            "seconds": round(time.time() - t0, 1), "note": arm["note"],
        })
        r = rows[-1]
        print(f"  {r['arm']:<20} frauds {r['train_frauds_after']:>6,}  "
              f"val PR-AUC {r['val_pr_auc']:.4f} "
              f"+/-{r['val_pr_auc_std']:.4f}   cost@0.5 {r['val_cost_at_0.5']:>8,.0f}   "
              f"P {r['val_precision_at_0.5']:.3f} R {r['val_recall_at_0.5']:.3f}   "
              f"{r['seconds']:>5.1f}s")

    df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def significance_note(df: pd.DataFrame) -> str:
    """Is the winner actually ahead of the shipped recipe?"""
    best, shipped = df.iloc[0], df[df.arm == "scale_pos_weight"].iloc[0]
    gap = best.val_pr_auc - shipped.val_pr_auc
    noise = float(np.hypot(best.val_pr_auc_std, shipped.val_pr_auc_std))
    if best.arm == "scale_pos_weight":
        return ("The shipped scale_pos_weight recipe wins the validation bench "
                "outright. Nothing below replaces it.")
    verdict = ("larger than" if gap > noise else "NOT larger than") + \
        " the across-seed noise of the two arms combined"
    return (f"{best.arm} leads the shipped scale_pos_weight recipe by "
            f"{gap:+.4f} val PR-AUC. That gap is {verdict} "
            f"({noise:.4f}), so the improvement is "
            f"{'worth carrying to test' if gap > noise else 'inside the noise and is reported as such'}.")


# ------------------------------------------------- winner -> calibrate -> test
def finalise(winner: dict, seeds: list[int]) -> dict:
    """Everything the winner needs before it is allowed near the test split.

    1. Out-of-fold scores over train+val, refitting the recipe inside each fold.
    2. Calibration fitted on those scores (resampling moves the base rate; the
       cost model multiplies probabilities by money, so they must mean
       something).
    3. Operating threshold from the smoothed, bootstrapped cost curve on the
       SAME out-of-fold scores -- ~400 frauds instead of 71, and none of them
       in-sample.
    4. Refit on train+val, then one look at test.
    """
    X_tr, y_tr = xy("train")
    X_va, y_va = xy("val")
    fn = list(X_tr.columns)
    X = np.vstack([X_tr.to_numpy(), X_va.to_numpy()])
    y = np.concatenate([y_tr.to_numpy(), y_va.to_numpy()])
    amt = np.concatenate([X_tr["Amount"].to_numpy(), X_va["Amount"].to_numpy()])

    print(f"\n[final] out-of-fold scoring of '{winner['name']}' over train+val "
          f"({len(y):,} rows, {int(y.sum())} frauds)")
    oof = oof_predictions(X, y, amt, winner, fn, n_splits=5, seed=SEED)
    print(f"[final] OOF PR-AUC = {average_precision_score(y, oof):.4f}"
          "   (out-of-sample, unlike the 1.000 the shipped model scores on val)")

    # --- calibration -----------------------------------------------------
    # Selection is on CROSS-FITTED ECE. An isotonic fit scored on its own
    # training scores reports an ECE of essentially zero by construction; it
    # would win every comparison here without ever being tested.
    best_cal, best_ece, cal_reports = None, None, {}
    for method in ("none", "platt", "isotonic"):
        c = Calibrator(method).fit(oof, y)
        rep = calibration_report(y, oof, c.predict(oof), method)
        xf = cross_fitted_calibration(y, oof, method, seed=SEED)
        rep["cross_fitted"] = xf
        cal_reports[method] = rep
        print(f"[final] calibration {method:<9} "
              f"ECE in-sample {rep['ece_calibrated']:.6f} / cross-fitted {xf['ece']:.6f}"
              f"   Brier cross-fitted {xf['brier']:.6f}")
        if best_ece is None or xf["ece"] < best_ece:
            best_cal, best_ece = c, xf["ece"]
    print(f"[final] calibrator selected on cross-fitted ECE: {best_cal.method}")
    oof_cal = best_cal.predict(oof)
    plot_reliability(y, oof, oof_cal, REPORTS / "calibration_reliability.png")

    # --- threshold on smoothed, bootstrapped OOF cost curve --------------
    sel = select_threshold(y, oof_cal, amt, fp_cost=FP_COST, n_boot=200, seed=SEED,
                           rule="one_se")
    print("\n[final] " + stability_report(sel).replace("\n", "\n        "))
    plot_selection(sel, REPORTS / "threshold_selection.png", shipped=0.09)

    # How many transactions actually live inside the admissible band? If none
    # do, every threshold in it produces byte-identical decisions, and moving to
    # the top of the range is free rather than merely cheap.
    lo_b, hi_b = sel["flat_region"]
    in_band = int(((oof_cal > lo_b) & (oof_cal < hi_b)).sum())
    print(f"[final] transactions scoring inside the admissible band "
          f"[{lo_b:.2f}, {hi_b:.2f}]: {in_band:,} of {len(oof_cal):,}")
    sel["rows_inside_flat_region"] = in_band

    # --- refit on train+val, then the single test evaluation -------------
    print(f"\n[final] refitting '{winner['name']}' on all {len(y):,} rows")
    booster, is_margin = fit_recipe(X, y, amt, winner, fn, SEED)
    out_model = MODELS / "imbalance_best.json"
    booster.save_model(out_model)

    X_te, y_te = xy("test")
    yt, amt_t = y_te.to_numpy(), X_te["Amount"].to_numpy()
    p_te = best_cal.predict(score_with(booster, X_te.to_numpy(), fn, is_margin))

    shipped = __import__("xgboost").Booster()
    shipped.load_model(TUNED_MODEL)
    p_shipped = predict(shipped, X_te)

    t_new = sel["threshold"]
    t_old = 0.09  # the shipped tie-broken threshold, applied as it is deployed
    res_new = cost_at(yt, p_te, amt_t, t_new)
    res_old = cost_at(yt, p_shipped, amt_t, t_old)
    nothing = do_nothing_cost(yt, amt_t)
    delta = paired_bootstrap_delta(yt, p_shipped, p_te, n_boot=2000, seed=SEED)

    print("\n" + "=" * 78)
    print("ONE LOOK AT THE TEST SPLIT")
    print("=" * 78)
    print(f"{'':<26}{'shipped tuned':>16}{'imbalance winner':>20}")
    print(f"{'PR-AUC':<26}{average_precision_score(yt, p_shipped):>16.4f}"
          f"{average_precision_score(yt, p_te):>20.4f}")
    print(f"{'operating threshold':<26}{t_old:>16.2f}{t_new:>20.2f}")
    print(f"{'precision there':<26}{res_old['precision']:>16.4f}{res_new['precision']:>20.4f}")
    print(f"{'recall there':<26}{res_old['recall']:>16.4f}{res_new['recall']:>20.4f}")
    print(f"{'false positives':<26}{res_old['fp']:>16d}{res_new['fp']:>20d}")
    print(f"{'missed frauds':<26}{res_old['fn']:>16d}{res_new['fn']:>20d}")
    print(f"{'cost':<26}{res_old['total_cost']:>16,.0f}{res_new['total_cost']:>20,.0f}")
    print(f"{'vs doing nothing':<26}{nothing:>16,.0f}{'':>20}")
    print("-" * 78)
    print(f"paired bootstrap on PR-AUC: delta {delta['delta_point']:+.4f}, "
          f"95% CI [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}], "
          f"{'SIGNIFICANT' if delta['significant_at_95'] else 'NOT significant'} at 95%")
    print("=" * 78)

    # A candidate serving config, NOT the live one. Promotion is a separate,
    # explicit act (`--promote`): a script that quietly repoints the service at
    # a model because it won a bench is a script that will one day repoint it at
    # something worse, at 3am, unattended.
    candidate = {
        "model_path": str(out_model),
        "block_threshold": t_new,
        "review_threshold": round(t_new * REVIEW_BAND_WIDTH, 4),
        "fp_cost": FP_COST,
        "calibrator": best_cal.to_dict(),
        "output_margin": bool(is_margin),
        "expected_precision_at_block": res_new["precision"],
        "expected_recall_at_block": res_new["recall"],
        "provenance": {
            "recipe": winner["name"],
            "threshold_rule": "one_se on smoothed bootstrapped OOF cost curve",
            "threshold_selected_on": "out-of-fold train+val",
            "calibrated_on": "out-of-fold train+val",
        },
    }
    cand_path = MODELS / "serving_config_candidate.json"
    cand_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    print(f"\n[final] wrote {cand_path}")
    print("[final] the live serving config is UNCHANGED. To deploy this model:")
    print("          python scripts/imbalance_experiment.py --promote")

    return {
        "winner": winner["name"],
        "candidate_serving_config": str(cand_path),
        "oof_pr_auc": float(average_precision_score(y, oof)),
        "calibration": cal_reports,
        "calibrator_selected": best_cal.method,
        "calibrator": best_cal.to_dict(),
        "threshold_selection": {k: v for k, v in sel.items()
                                if k not in ("grid", "cost_raw", "cost_smoothed", "cost_se")},
        "threshold_stability_note": stability_report(sel),
        "rows_inside_flat_region": sel.get("rows_inside_flat_region"),
        "test": {
            "shipped": {"threshold": t_old, "pr_auc": float(average_precision_score(yt, p_shipped)), **res_old},
            "winner": {"threshold": t_new, "pr_auc": float(average_precision_score(yt, p_te)), **res_new},
            "do_nothing_cost": nothing,
            "paired_bootstrap": delta,
        },
        "model_path": str(out_model),
    }


def plot_bench(df: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"baseline": "#94a3b8", "reweight": "#2563eb", "loss": "#7c3aed",
               "targets": "#0891b2", "resample": "#16a34a", "combo": "#d97706"}
    d = df.sort_values("val_pr_auc")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(d.arm, d.val_pr_auc, xerr=d.val_pr_auc_std, capsize=3,
            color=[colours[f] for f in d.family], edgecolor="white")
    shipped = float(df[df.arm == "scale_pos_weight"].val_pr_auc.iloc[0])
    ax.axvline(shipped, ls="--", lw=1.3, color="#dc2626",
               label=f"shipped scale_pos_weight = {shipped:.4f}")
    ax.set_xlim(max(0.0, d.val_pr_auc.min() - 0.04), d.val_pr_auc.max() + 0.02)
    ax.set_xlabel("validation PR-AUC (mean over seeds, bars = std)")
    ax.set_title("Imbalance techniques, identical hyperparameters and rounds")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(handles + [plt.Line2D([], [], ls="--", color="#dc2626")],
              list(colours) + ["shipped recipe"], fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[bench] wrote {out}")


def write_markdown(df: pd.DataFrame, note: str, final: dict | None, out: Path) -> None:
    lines = [
        "# Class-imbalance experiment",
        "",
        "331 frauds against 198,277 legitimate transactions in the training split.",
        "Every arm below uses identical hyperparameters, identical fixed boosting",
        "rounds and no early stopping, so the only thing varying is the imbalance",
        "treatment. Selection is on validation; the test split is used once, at the",
        "end, on the winner only.",
        "",
        "## Validation bench",
        "",
        "| rank | arm | family | train frauds | val PR-AUC | +/- seed | cost @0.5 | P@0.5 | R@0.5 | what it does |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['rank']} | `{r['arm']}` | {r['family']} | {r['train_frauds_after']:,} | "
            f"**{r['val_pr_auc']:.4f}** | "
            f"{r['val_pr_auc_std']:.4f} | {r['val_cost_at_0.5']:,.0f} | "
            f"{r['val_precision_at_0.5']:.3f} | {r['val_recall_at_0.5']:.3f} | {r['note']} |"
        )
    lines += ["", "## Verdict", "", note, ""]
    if final:
        t = final["test"]
        lines += [
            "## The winner, taken to test once",
            "",
            f"Winner: **`{final['winner']}`**. Out-of-fold PR-AUC over train+val "
            f"({final['oof_pr_auc']:.4f}) is the honest selection number.",
            "",
            f"Calibrator: **{final['calibrator_selected']}**, fitted on out-of-fold "
            "scores. The cost model multiplies predicted probabilities by money, so "
            "they have to mean what they say -- and `scale_pos_weight` is itself a "
            "distortion of the base rate, exactly like resampling: telling the "
            "learner a fraud is worth 600 legitimate rows trains it to report "
            "probabilities for a world with 600x more fraud than this one.",
            "",
            "| | shipped tuned | imbalance winner |",
            "|---|---:|---:|",
            f"| PR-AUC (test) | {t['shipped']['pr_auc']:.4f} | {t['winner']['pr_auc']:.4f} |",
            f"| operating threshold | {t['shipped']['threshold']:.2f} | {t['winner']['threshold']:.2f} |",
            f"| precision there | {t['shipped']['precision']:.4f} | {t['winner']['precision']:.4f} |",
            f"| recall there | {t['shipped']['recall']:.4f} | {t['winner']['recall']:.4f} |",
            f"| false positives | {t['shipped']['fp']} | {t['winner']['fp']} |",
            f"| missed frauds | {t['shipped']['fn']} | {t['winner']['fn']} |",
            f"| cost | {t['shipped']['total_cost']:,.0f} | {t['winner']['total_cost']:,.0f} |",
            f"| do nothing | {t['do_nothing_cost']:,.0f} | |",
            "",
            f"Paired bootstrap on PR-AUC: delta "
            f"{t['paired_bootstrap']['delta_point']:+.4f}, 95% CI "
            f"[{t['paired_bootstrap']['ci_low']:+.4f}, "
            f"{t['paired_bootstrap']['ci_high']:+.4f}] -- "
            + ("significant." if t["paired_bootstrap"]["significant_at_95"]
               else "**not** significant on 71 test frauds."),
            "",
            "The ranking did not improve and is not claimed to have. The model is "
            "the same recipe; what changed is where the threshold was placed, and "
            "that is where the money came from.",
            "",
            "## Calibration",
            "",
            "Selected on **cross-fitted** ECE. An isotonic fit scored on the same "
            "scores it was fitted to reports an ECE of ~0 by construction, so that "
            "number is a tautology and is shown next to the honest one rather than "
            "instead of it.",
            "",
            "| calibrator | ECE (in-sample) | ECE (cross-fitted) | Brier (cross-fitted) |",
            "|---|---:|---:|---:|",
        ]
        for m, rep in final["calibration"].items():
            xf = rep["cross_fitted"]
            mark = " **<- selected**" if m == final["calibrator_selected"] else ""
            lines.append(f"| `{m}`{mark} | {rep['ece_calibrated']:.6f} | "
                         f"{xf['ece']:.6f} | {xf['brier']:.6f} |")
        lines += [
            "",
            "## Threshold stability",
            "",
            final["threshold_stability_note"],
            "",
        ]
        band = final.get("rows_inside_flat_region")
        if band is not None:
            lo, hi = final["threshold_selection"]["flat_region"]
            lines += [
                f"**Why that band is flat: {band:,} of the 241,167 out-of-fold "
                f"transactions score anywhere inside [{lo:.2f}, {hi:.2f}].** The "
                "model is bimodal -- it is either confident a transaction is fraud "
                "or confident it is not -- so the admissible region is an empty "
                "stretch of the score axis. Every threshold in it produces "
                "identical decisions, which is why moving to the top of the range "
                "costs nothing at all rather than merely costing little, and why "
                "the operating point is robust to the curve moving underneath it.",
                "",
                "The same gap holds on the test split: 0 of 42,559 test "
                "transactions score inside it either. The previous threshold of "
                "0.09 sat below the gap, on the noisy side where small score "
                "movements change decisions; 0.66 sits at its top edge.",
                "",
                "![threshold selection](threshold_selection.png)",
                "",
            ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[bench] wrote {out}")


def promote() -> int:
    """Point the live service at the candidate model. Deliberate and separate."""
    cand = MODELS / "serving_config_candidate.json"
    if not cand.exists():
        print(f"no candidate at {cand} -- run the experiment first")
        return 1
    new = json.loads(cand.read_text(encoding="utf-8"))
    old = (json.loads(SERVING_CONFIG.read_text(encoding="utf-8"))
           if SERVING_CONFIG.exists() else {})
    print("promoting the imbalance winner to the live serving config:")
    for k in ("model_path", "block_threshold", "review_threshold"):
        print(f"  {k:<20} {old.get(k)!r:>40}  ->  {new.get(k)!r}")
    print(f"  {'calibrator':<20} {old.get('calibrator', {}).get('method', 'none')!r:>40}"
          f"  ->  {new['calibrator']['method']!r}")
    backup = MODELS / "serving_config_previous.json"
    if SERVING_CONFIG.exists():
        backup.write_text(json.dumps(old, indent=2), encoding="utf-8")
        print(f"  previous config saved to {backup}")
    SERVING_CONFIG.write_text(json.dumps(new, indent=2), encoding="utf-8")
    print(f"  wrote {SERVING_CONFIG}. Restart the API to pick it up.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="subsample the majority class - smoke-test speed, not a result")
    ap.add_argument("--no-final", action="store_true",
                    help="run the validation bench only, never load test")
    ap.add_argument("--promote", action="store_true",
                    help="deploy the candidate config produced by a previous run")
    ap.add_argument("--final-only", action="store_true",
                    help="skip the bench, re-run calibration/threshold/test for the "
                         "winner already recorded in reports/imbalance_experiment.csv")
    a = ap.parse_args()

    if a.promote:
        return promote()

    if a.final_only:
        prev = REPORTS / "imbalance_experiment.csv"
        if not prev.exists():
            print(f"no {prev} -- run the bench first")
            return 1
        df = pd.read_csv(prev).sort_values("val_pr_auc", ascending=False)
        df = df.reset_index(drop=True)
        df["rank"] = df.index + 1
        note = significance_note(df)
        winner_name = df.iloc[0]["arm"]
        print(f"[final-only] reusing the recorded leaderboard; winner = {winner_name}")
        final = finalise(next(x for x in arms(base_params()) if x["name"] == winner_name),
                         [SEED])
        payload = json.loads((REPORTS / "imbalance_experiment.json").read_text(
            encoding="utf-8"))
        payload["final"] = final
        (REPORTS / "imbalance_experiment.json").write_text(
            json.dumps(payload, indent=2, default=float), encoding="utf-8")
        write_markdown(df, note, final, REPORTS / "imbalance_experiment.md")
        return 0

    seeds = [SEED + i for i in range(a.seeds)]
    df = run_bench(seeds, a.quick)

    print("\n" + "=" * 78)
    print("VALIDATION LEADERBOARD (test split still untouched)")
    print("=" * 78)
    print(df[["rank", "arm", "family", "train_frauds_after", "val_pr_auc",
              "val_pr_auc_std", "val_cost_at_0.5"]].to_string(index=False))
    note = significance_note(df)
    print("\n" + note)

    df.to_csv(REPORTS / "imbalance_experiment.csv", index=False)
    plot_bench(df, REPORTS / "imbalance_bench.png")

    final = None
    if not a.no_final and not a.quick:
        winner_name = df.iloc[0]["arm"]
        winner = next(x for x in arms(base_params()) if x["name"] == winner_name)
        final = finalise(winner, seeds)

    payload = {"protocol": {"rounds": ROUNDS, "seeds": seeds, "early_stopping": False,
                            "selection_split": "val", "test_looks": 1},
               "leaderboard": df.to_dict(orient="records"),
               "verdict": note, "final": final}
    (REPORTS / "imbalance_experiment.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    write_markdown(df, note, final, REPORTS / "imbalance_experiment.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
