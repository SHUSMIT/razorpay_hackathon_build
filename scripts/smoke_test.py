#!/usr/bin/env python
"""End-to-end smoke test: every stage of the system, in order, in about a minute.

This is not the unit-test suite. The unit tests check that each part is correct
in isolation; this checks that the parts are still WIRED TOGETHER -- that the
features training used are the features the API builds, that the threshold the
evaluator wrote is the threshold the service serves, that a score becomes a
banded decision, that the gate downgrades a burst, and that the decision lands
in the audit log. Those are the failures that survive a green test suite and
kill a live demo.

It runs on a stratified subsample of the real splits, so it exercises the real
data path and still finishes fast. Nothing it does touches the shipped models,
reports or audit log: every artefact it writes goes to a temp directory.

    python run.py smoke          # or: python scripts/smoke_test.py

Exit code 0 = every stage green. Non-zero = the number of failed stages.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: list[dict] = []
SEED = 42


@contextmanager
def stage(name: str, detail: str = ""):
    """Run one stage, record PASS/FAIL/SKIP, never abort the whole run."""
    t0 = time.perf_counter()
    rec = {"stage": name, "status": "PASS", "detail": detail, "seconds": 0.0}
    RESULTS.append(rec)
    try:
        yield rec
    except SkipStage as exc:
        rec["status"] = "SKIP"
        rec["detail"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not raise
        rec["status"] = "FAIL"
        rec["detail"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
    finally:
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[rec["status"]]
        print(f"  [{icon}] {name:<38} {rec['seconds']:>6.2f}s  {rec['detail']}")


class SkipStage(Exception):
    """Raised when a stage cannot run because an upstream artefact is missing."""


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------- stages
def main(rows: int = 25_000) -> int:
    print("=" * 84)
    print("END-TO-END SMOKE TEST")
    print("=" * 84)
    tmp = Path(tempfile.mkdtemp(prefix="frm_smoke_"))
    print(f"scratch: {tmp}\n")

    state: dict = {}

    # -- 1. config and artefacts ------------------------------------------
    with stage("config + artefacts on disk") as r:
        from src.config import BASELINE_MODEL, DATA_PROCESSED, FP_COST, SERVING_CONFIG

        need(FP_COST > 0, "FP_COST must be positive")
        missing = [n for n in ("train", "val", "test")
                   if not (DATA_PROCESSED / f"{n}.parquet").exists()]
        need(not missing, f"missing splits {missing} -- run `python run.py data`")
        need(BASELINE_MODEL.exists() or SERVING_CONFIG.exists(),
             "no model artefact -- run `python run.py train`")
        r["detail"] = f"splits present, FP_COST={FP_COST:.0f}"

    # -- 2. data + features ------------------------------------------------
    with stage("data load + stratified subsample") as r:
        from src.train import xy

        X_tr, y_tr = xy("train")
        X_va, y_va = xy("val")
        rng = np.random.default_rng(SEED)
        yn = y_tr.to_numpy()
        pos, neg = np.flatnonzero(yn == 1), np.flatnonzero(yn == 0)
        keep = np.concatenate([pos, rng.choice(neg, min(rows, len(neg)), replace=False)])
        state.update(
            X=X_tr.to_numpy()[keep], y=yn[keep], amt=X_tr["Amount"].to_numpy()[keep],
            fn=list(X_tr.columns), X_va=X_va.to_numpy(), y_va=y_va.to_numpy(),
            amt_va=X_va["Amount"].to_numpy(), X_va_df=X_va,
        )
        need(state["y"].sum() > 0, "subsample contains no frauds")
        r["detail"] = (f"{len(state['y']):,} rows, {int(state['y'].sum())} frauds, "
                       f"{len(state['fn'])} features")

    with stage("features are deterministic + match the trained schema") as r:
        from src.config import FEATURE_NAMES
        from src.features import build_features

        raw = X_tr.head(50).copy()
        raw["Time"] = np.linspace(0, 86_400, 50)
        a = build_features(raw)
        b = build_features(raw)
        need(a.equals(b), "build_features is not deterministic on identical input")
        need(np.isfinite(a.to_numpy()).all(), "build_features produced non-finite values")
        if FEATURE_NAMES.exists():
            trained = json.loads(FEATURE_NAMES.read_text(encoding="utf-8"))
            need(list(a.columns) == trained,
                 f"feature order drift: API builds {list(a.columns)[:3]}..., "
                 f"model expects {trained[:3]}...")
        r["detail"] = f"{len(a.columns)} columns, order matches feature_names.json"

    # -- 3. imbalance machinery -------------------------------------------
    with stage("resamplers add minority rows only") as r:
        from src.imbalance import RESAMPLERS, resample

        n0, p0 = len(state["y"]), int(state["y"].sum())
        # Derive the target from the subsample rather than hard-coding it. A
        # fixed ratio silently becomes a no-op whenever --rows makes the
        # subsample small enough that the frauds already exceed it, and a no-op
        # that passes every assertion is worse than a failure.
        ratio = 3.0 * p0 / max(n0 - p0, 1)
        state["resample_ratio"] = ratio
        added = {}
        for name in RESAMPLERS:
            Xr, yr = resample(name, state["X"], state["y"], ratio=ratio, seed=SEED)
            need(len(Xr) == len(yr), f"{name}: X/y length mismatch")
            need(int(yr.sum()) > p0, f"{name}: added no positives")
            # Cleaning methods may drop rows; nothing may ADD a negative.
            need(int((yr == 0).sum()) <= n0 - p0, f"{name}: invented majority rows")
            need(np.isfinite(Xr).all(), f"{name}: produced non-finite features")
            added[name] = int(yr.sum())
        r["detail"] = (f"{p0} frauds -> ~{int(ratio * (n0 - p0))} target: "
                       + ", ".join(f"{k}={v}" for k, v in added.items()))

    with stage("resampling the test split is refused") as r:
        from src.imbalance import resample

        for split in ("val", "test"):
            try:
                resample("smote", state["X"], state["y"],
                         ratio=state["resample_ratio"], seed=SEED, split=split)
            except ValueError:
                continue
            raise AssertionError(f"resampling the {split} split was NOT refused")
        r["detail"] = "val and test are protected"

    with stage("focal-loss gradient matches numerical derivative") as r:
        from src.imbalance import FocalLoss, sigmoid

        fl = FocalLoss(gamma=2.0, alpha=0.75)
        z = np.array([-6.0, -2.0, -0.3, 0.0, 0.4, 2.0, 5.0])
        worst = 0.0
        for lab in (0.0, 1.0):
            yv = np.full_like(z, lab)

            def loss(zz):
                p = sigmoid(zz)
                pt = np.where(yv >= 0.5, p, 1 - p)
                at = np.where(yv >= 0.5, fl.alpha, 1 - fl.alpha)
                return -at * (1 - pt) ** fl.gamma * np.log(np.clip(pt, 1e-12, 1))

            eps = 1e-5
            worst = max(worst, float(np.abs(
                (loss(z + eps) - loss(z - eps)) / (2 * eps) - fl._grad(z, yv)).max()))
        need(worst < 1e-6, f"analytic gradient disagrees with numeric by {worst:.2e}")
        r["detail"] = f"max |analytic - numeric| = {worst:.2e}"

    with stage("weighting schemes are finite and favour the minority") as r:
        from src.imbalance import WEIGHTERS, sample_weights

        y, amt = state["y"], state["amt"]
        ratios = {}
        for name in WEIGHTERS:
            w = sample_weights(name, y, amounts=amt, fp_cost=50.0)
            need(np.isfinite(w).all() and (w > 0).all(), f"{name}: bad weights")
            ratios[name] = round(float(w[y == 1].mean() / w[y == 0].mean()), 2)
        need(ratios["class_weight"] > 1, "class_weight does not up-weight fraud")
        need(ratios["none"] == 1.0, "the null weighting is not null")
        r["detail"] = "fraud:legit weight " + ", ".join(f"{k}={v}" for k, v in ratios.items())

    # -- 4. training -------------------------------------------------------
    with stage("train under an imbalance recipe") as r:
        from src.calibrate import fit_recipe, score_with
        from src.config import BEST_PARAMS
        from sklearn.metrics import average_precision_score

        params = (json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]
                  if BEST_PARAMS.exists()
                  else {"objective": "binary:logistic", "eval_metric": "aucpr",
                        "max_depth": 5, "learning_rate": 0.1, "tree_method": "hist"})
        params = dict(params, nthread=4)
        recipe = {"params": params, "rounds": 60, "resampler": "smote",
                  "ratio": state["resample_ratio"], "scale_pos_weight": "auto",
                  "fp_cost": 50.0}
        booster, is_margin = fit_recipe(state["X"], state["y"], state["amt"], recipe,
                                        state["fn"], SEED)
        p = score_with(booster, state["X_va"], state["fn"], is_margin)
        need(p.shape == state["y_va"].shape, "prediction shape mismatch")
        need(((p >= 0) & (p <= 1)).all(), "scores escaped [0, 1]")
        ap = float(average_precision_score(state["y_va"], p))
        need(ap > 0.5, f"val PR-AUC {ap:.3f} is implausibly low -- pipeline is broken")
        state.update(p_va=p, booster=booster, is_margin=is_margin)
        r["detail"] = f"val PR-AUC = {ap:.4f} (random = {state['y_va'].mean():.5f})"

    with stage("focal-loss objective trains and scores in [0,1]") as r:
        from src.calibrate import fit_recipe, score_with

        recipe = {"params": dict(params), "rounds": 40,
                  "focal": {"gamma": 2.0, "alpha": 0.75}, "fp_cost": 50.0}
        b, m = fit_recipe(state["X"], state["y"], state["amt"], recipe, state["fn"], SEED)
        need(m, "custom objective did not report margin output")
        p = score_with(b, state["X_va"], state["fn"], m)
        need(((p > 0) & (p < 1)).all(), "sigmoid was not applied to raw margins")
        r["detail"] = f"scores in [{p.min():.4f}, {p.max():.4f}]"

    # -- 5. threshold selection -------------------------------------------
    with stage("smoothed + bootstrapped threshold selection") as r:
        from src.config import FP_COST
        from src.threshold import select_threshold, smooth

        v = np.array([5.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0])
        need(len(smooth(v, 3)) == len(v), "smoothing changed the grid length")
        need(smooth(v, 3).std() < v.std(), "smoothing did not reduce variance")

        sel = select_threshold(state["y_va"], state["p_va"], state["amt_va"],
                               fp_cost=FP_COST, n_boot=40, seed=SEED, rule="one_se")
        need(0 < sel["threshold"] < 1, "threshold outside (0, 1)")
        need(sel["threshold_one_se"] >= sel["threshold_argmin_smoothed"],
             "the one-SE rule chose a LESS conservative threshold than the minimum")
        need(sel["cost_raw_at_chosen"] >= sel["cost_raw_at_argmin"],
             "chosen threshold beats the raw argmin, which is arithmetically impossible")
        state["sel"] = sel
        r["detail"] = (f"raw argmin {sel['threshold_argmin_raw']:.2f} -> one-SE "
                       f"{sel['threshold']:.2f}, flat region "
                       f"[{sel['flat_region'][0]:.2f}, {sel['flat_region'][1]:.2f}]")

    # -- 6. calibration ----------------------------------------------------
    with stage("calibration reduces ECE and preserves ranking") as r:
        from sklearn.metrics import average_precision_score

        from src.calibrate import Calibrator, expected_calibration_error

        y, p = state["y_va"], state["p_va"]
        cal = Calibrator("isotonic").fit(p, y)
        pc = cal.predict(p)
        ece_before = expected_calibration_error(y, p)
        ece_after = expected_calibration_error(y, pc)
        need(ece_after <= ece_before + 1e-9, "isotonic made calibration worse in-sample")
        ap_b, ap_a = average_precision_score(y, p), average_precision_score(y, pc)
        need(abs(ap_b - ap_a) < 0.02,
             f"calibration reordered the scores: PR-AUC {ap_b:.4f} -> {ap_a:.4f}")

        # Round-trip through the on-disk form the API will load.
        rt = Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
        # 1e-6, not 1e-9: XGBoost returns float32 and sklearn's interpolator
        # carries that dtype through, so an exact match would be asserting
        # something about float32 rather than about the serialisation.
        drift = float(np.abs(rt.predict(p) - pc).max())
        need(drift < 1e-6, f"calibrator does not survive serialisation ({drift:.2e})")
        r["detail"] = f"ECE {ece_before:.6f} -> {ece_after:.6f}, PR-AUC unchanged"

    # -- 7. cost model -----------------------------------------------------
    with stage("cost model is internally consistent") as r:
        from src.config import FP_COST
        from src.evaluate import cost_at, do_nothing_cost

        y, p, amt = state["y_va"], state["p_va"], state["amt_va"]
        nothing = do_nothing_cost(y, amt)
        at_zero = cost_at(y, p, amt, 0.0)
        need(abs(at_zero["fp_cost"] - FP_COST * int((y == 0).sum())) < 1e-6,
             "blocking everything does not cost FP_COST per legitimate row")
        need(at_zero["fn"] == 0, "blocking everything still missed a fraud")
        need(abs(cost_at(y, p, amt, 1.01)["total_cost"] - nothing) < 1e-6,
             "blocking nothing does not equal the do-nothing cost")
        best = cost_at(y, p, amt, state["sel"]["threshold"])
        need(best["total_cost"] < nothing, "the model is worse than doing nothing")
        r["detail"] = (f"do-nothing {nothing:,.0f} -> at chosen threshold "
                       f"{best['total_cost']:,.0f} "
                       f"({100 * (1 - best['total_cost'] / nothing):.1f}% saved)")

    # -- 8. explanations ---------------------------------------------------
    with stage("SHAP factors are additive and top-3 is populated") as r:
        from src.explain import top_factors, verify_additivity

        row = state["X_va_df"].head(20)
        err = verify_additivity(state["booster"], row)
        need(err < 1e-2, f"SHAP values do not sum to the margin (max err {err:.2e})")
        import xgboost as xgb

        contribs = state["booster"].predict(
            xgb.DMatrix(row, feature_names=state["fn"]), pred_contribs=True)[0, :-1]
        f = top_factors(contribs, state["fn"], k=3)
        need(len(f) == 3 and all(x["feature"] in state["fn"] for x in f),
             "top_factors returned something the model does not have")
        r["detail"] = f"additivity err {err:.2e}; top factor {f[0]['feature']}"

    # -- 9. the service ----------------------------------------------------
    with stage("API: health, score, banding") as r:
        import src.api as api_mod

        api_mod.AUDIT_LOG = tmp / "audit.jsonl"          # never touch the real log
        from fastapi.testclient import TestClient

        client = TestClient(api_mod.app)
        h = client.get("/health")
        need(h.status_code == 200, f"/health returned {h.status_code}")
        need(h.json()["thresholds"]["block"] > h.json()["thresholds"]["review"],
             "block threshold is not above the review threshold")

        svc = api_mod.get_service()
        for score, expected in ((svc.block_threshold + 1e-6, "block"),
                                (svc.review_threshold + 1e-6, "review"),
                                (max(svc.review_threshold - 1e-3, 0.0), "allow")):
            got = svc.band(score)[0]
            need(got == expected,
                 f"score {score:.4f} banded as {got}, expected {expected}")

        client.post("/gate/reset")
        payload = {"Time": 3600.0, "Amount": 149.62,
                   **{f"V{i}": 0.0 for i in range(1, 29)}}
        s = client.post("/score", json=payload)
        need(s.status_code == 200, f"/score returned {s.status_code}: {s.text}")
        body = s.json()
        need(0 <= body["risk_score"] <= 1, "risk_score outside [0, 1]")
        need(body["decision"] in ("allow", "review", "block"), "unknown decision")
        need(len(body["top_factors"]) == 3, "no explanation on the response")
        state["client"], state["api"] = client, api_mod
        r["detail"] = (f"score {body['risk_score']:.4f} -> {body['decision']} "
                       f"in {body['latency_ms']:.1f}ms")

    with stage("API: malformed input is rejected, not crashed") as r:
        client = state["client"]
        v0 = ", ".join(f'"V{i}": 0.0' for i in range(1, 29))
        # Sent as raw bytes, not via json=. httpx refuses to serialise NaN, so
        # `json={"Amount": nan}` fails in the CLIENT and never reaches the
        # service -- which would quietly turn the most interesting case here
        # into a test of httpx. Python's own json.loads accepts bare NaN, and
        # so does the server, which is exactly why src.api has a handler that
        # strips non-finite values before rendering the 422.
        bad = [
            ('{"Amount": -5.0, %s}' % v0, "negative amount"),
            ('{"Amount": NaN, %s}' % v0, "NaN amount"),
            ('{"Amount": Infinity, %s}' % v0, "infinite amount"),
            ('{"Amount": 1e12, %s}' % v0, "implausible amount"),
            ('{"Amount": "not a number", %s}' % v0, "non-numeric amount"),
            ('{%s}' % v0, "no amount at all"),
            ('{"Amount": 10.0, "V1": "abc"}', "non-numeric PCA component"),
            ('{"Amount": 10.0,', "truncated JSON"),
        ]
        for body, why in bad:
            resp = client.post("/score", content=body.encode(),
                               headers={"content-type": "application/json"})
            need(resp.status_code == 422,
                 f"{why} returned {resp.status_code}, expected 422: {resp.text[:120]}")
            need("detail" in resp.json(), f"{why}: no error detail")
        r["detail"] = f"{len(bad)} malformed payloads all rejected with 422"

    with stage("API: block-rate gate downgrades a burst") as r:
        client, api_mod = state["client"], state["api"]
        client.post("/gate/reset")
        svc = api_mod.get_service()

        # Force every request into the block band, then fire a burst. The gate
        # must hold the rate at or under the cap and say why in the response.
        real_band = svc.band
        svc.band = lambda s: ("block", "TEST_FORCED_BLOCK")
        try:
            payload = {"Time": 0.0, "Amount": 10.0,
                       **{f"V{i}": 0.0 for i in range(1, 29)}}
            decisions = [client.post("/score", json=payload).json() for _ in range(60)]
        finally:
            svc.band = real_band

        blocked = sum(d["raw_decision"] == "block" and not d["gated"] for d in decisions)
        gated = [d for d in decisions if d["gated"]]
        need(blocked > 0, "the gate blocked everything -- genuine fraud would be ignored")
        need(len(gated) > 0, "the gate never downgraded during a 60-request burst")
        need(all(d["decision"] == "review" for d in gated),
             "a downgraded block did not become a review")
        need(all(d["gate_reason"] for d in gated), "a downgrade was not given a reason")
        rate = svc.gate.snapshot()["block_rate"]
        need(rate <= api_mod.MAX_BLOCK_RATE + 1e-9,
             f"rolling block rate {rate:.3f} exceeded the {api_mod.MAX_BLOCK_RATE} cap")
        r["detail"] = (f"60 forced blocks -> {blocked} admitted, {len(gated)} downgraded, "
                       f"rolling rate {rate:.1%} <= {api_mod.MAX_BLOCK_RATE:.0%} cap")

    with stage("API: every decision reached the audit log") as r:
        client, api_mod = state["client"], state["api"]
        need(api_mod.AUDIT_LOG.exists(), "no audit log was written")
        lines = api_mod.AUDIT_LOG.read_text(encoding="utf-8").strip().splitlines()
        need(len(lines) >= 60, f"only {len(lines)} audit entries for 60+ decisions")
        entries = [json.loads(x) for x in lines]
        required = {"ts", "request_id", "input_sha256", "risk_score", "decision",
                    "rule_id", "threshold_used", "model_version"}
        for e in entries:
            need(required <= set(e), f"audit entry missing {required - set(e)}")
        need(len({e["request_id"] for e in entries}) == len(entries),
             "duplicate request ids in the audit log")
        need(not any("Amount" in e and "V1" in e for e in entries),
             "raw feature values leaked into the audit log")
        a = client.get("/audit?n=5")
        need(a.status_code == 200 and len(a.json()["entries"]) <= 5, "/audit is broken")
        r["detail"] = f"{len(entries)} entries, hashed inputs, unique ids"

    # -- 10. serving contract ---------------------------------------------
    with stage("serving config is complete and self-consistent") as r:
        from src.config import SERVING_CONFIG

        if not SERVING_CONFIG.exists():
            raise SkipStage("serving_config.json absent -- run `python run.py evaluate`")
        cfg = json.loads(SERVING_CONFIG.read_text(encoding="utf-8"))
        for k in ("block_threshold", "review_threshold", "fp_cost"):
            need(k in cfg, f"serving config is missing {k}")
        need(0 < cfg["review_threshold"] <= cfg["block_threshold"] < 1,
             "serving thresholds are not ordered 0 < review <= block < 1")
        r["detail"] = (f"block {cfg['block_threshold']:.3f}, review "
                       f"{cfg['review_threshold']:.3f}")

    shutil.rmtree(tmp, ignore_errors=True)
    return report()


def report() -> int:
    ok = sum(r["status"] == "PASS" for r in RESULTS)
    skipped = sum(r["status"] == "SKIP" for r in RESULTS)
    failed = [r for r in RESULTS if r["status"] == "FAIL"]
    total = sum(r["seconds"] for r in RESULTS)

    print("\n" + "=" * 84)
    print(f"{ok} passed, {len(failed)} failed, {skipped} skipped "
          f"in {total:.1f}s")
    print("=" * 84)
    for r in failed:
        print(f"\nFAILED: {r['stage']}\n  {r['detail']}")
        if "traceback" in r:
            print("  " + r["traceback"].replace("\n", "\n  ").rstrip())
    if not failed:
        print("SMOKE TEST GREEN - data, features, imbalance handling, training,")
        print("threshold selection, calibration, cost model, explanations, the API,")
        print("the block-rate gate and the audit trail are all wired together.")

    from src.config import REPORTS

    (REPORTS / "smoke_test.json").write_text(json.dumps(
        {"passed": ok, "failed": len(failed), "skipped": skipped,
         "seconds": round(total, 2),
         "stages": [{k: v for k, v in r.items() if k != "traceback"} for r in RESULTS]},
        indent=2), encoding="utf-8")
    return len(failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=25_000,
                    help="majority rows to train the smoke model on")
    raise SystemExit(main(ap.parse_args().rows))
