"""FastAPI risk-scoring service.

Defense-only. The service scores a transaction and returns a bounded decision:

  allow / review / block

Four things make this more than a model behind HTTP:

  1. THRESHOLDS COME FROM THE COST CURVE, not from 0.5. The block threshold is
     whatever minimised money lost, selected on a slice of validation that the
     models, the blend weights and the calibrator never touched. The review
     band beneath it is sized by how much traffic a human team can actually
     look at, not by a magic constant.
  2. THE BLOCK RATE IS GATED. The service will never auto-block more than
     MAX_BLOCK_RATE of a rolling window. If a burst would push it over, block
     decisions are force-downgraded to `review` and the downgrade is logged
     with its reason. Drift, an attack or a bad deploy degrades this into a
     review queue rather than a merchant outage.
  3. EVERY DECISION IS EXPLAINED IN HUMAN TERMS. The features are named, so a
     reviewer is told "merchant category: Cruise Lines - 67% of transactions in
     this group were fraud, against 0.16% overall" rather than being handed a
     SHAP value for an anonymous component.
  4. EVERY DECISION IS AUDITED, and anything sent to a human lands in a review
     queue that a person actually resolves.

Real traffic is incomplete, so every field except the amount is optional. What
could not be derived is reported back in `missing_fields` rather than being
silently invented.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.calibration import Calibration
from src.config import (
    AUDIT_LOG,
    GATE_MIN_SAMPLE,
    MAX_BLOCK_RATE,
    MODELS,
    REVIEW_QUEUE,
    ROLLING_WINDOW,
    SERVING_CONFIG,
)
from src.context import FeatureContext
from src.models import ensemble_predict, load, predict
from src.schema import CATEGORICAL, FEATURES, label
from src.serve_features import ServingFeaturizer

Decision = Literal["allow", "review", "block"]
DEFAULT_MODEL_VERSION = "frm-1.0.0"


# --------------------------------------------------------------------- schema
class Transaction(BaseModel):
    """One transaction, as a payment gateway would send it.

    Only `amount` is required. Everything else is optional because real traffic
    arrives incomplete, and the service reports what it could not derive.
    """

    amount: float = Field(..., description="transaction amount")
    date: str | None = Field(None, description="ISO timestamp, for time-of-day")
    use_chip: str | None = Field(None, description="Chip / Swipe / Online Transaction")
    merchant_state: str | None = None
    zip: float | None = Field(None, description="absent implies card-not-present")
    mcc: int | None = Field(None, description="merchant category code")
    mcc_category: str | None = Field(None, description="or the category name directly")
    errors: str | None = Field(None, description="terminal error, if any")
    credit_limit: float | None = None
    card_brand: str | None = None
    card_type: str | None = None
    has_chip: bool | None = None
    is_online: bool | None = None

    model_config = {"extra": "allow"}

    @field_validator("amount")
    @classmethod
    def finite_amount(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("amount must be a finite number")
        if abs(v) > 1e9:
            raise ValueError("amount implausibly large (>1e9); rejected as malformed")
        return v


class ScoreResponse(BaseModel):
    request_id: str
    risk_score: float
    risk_percent: str
    decision: Decision
    raw_decision: Decision
    gated: bool
    gate_reason: str | None
    top_factors: list[dict]
    rule_id: str
    thresholds: dict
    model_version: str
    scored_by: str
    explained_by: str
    missing_fields: list[str]
    latency_ms: float
    base_rate: dict | None = None


class ReviewResolution(BaseModel):
    request_id: str
    action: Literal["approve", "decline"]
    reviewer: str = Field("demo-reviewer", max_length=64)
    note: str = Field("", max_length=500)


# ------------------------------------------------------------------ gate state
class BlockRateGate:
    """Rolling-window cap on the automatic block rate."""

    def __init__(self, window: int = ROLLING_WINDOW, max_rate: float = MAX_BLOCK_RATE,
                 min_sample: int = GATE_MIN_SAMPLE):
        self.window, self.max_rate, self.min_sample = window, max_rate, min_sample
        self.recent: deque[int] = deque(maxlen=window)
        self.lock = threading.Lock()

    def current_rate(self) -> float:
        """Raw observed rate over the decisions actually seen."""
        return (sum(self.recent) / len(self.recent)) if self.recent else 0.0

    def enforced_rate(self) -> float:
        """The rate the gate compares against the cap.

        Same denominator as `admit`: max(observed, min_sample). Reporting the
        raw rate is misleading on a cold window -- three decisions of which one
        is a block reads as "33% against a 5% cap", which looks like the safety
        system has failed when it is in fact holding at 1/20 = 5%.
        """
        if not self.recent:
            return 0.0
        return sum(self.recent) / max(len(self.recent), self.min_sample)

    def admit(self, wants_block: bool) -> tuple[bool, str | None]:
        """Decide whether a block is allowed through. Returns (allowed, reason).

        The rate is measured against max(observed, min_sample) decisions, not
        the observed count alone. If a short window were exempt, a cold service
        could auto-block its first `min_sample` requests outright and blow
        straight through the bound -- precisely the burst the gate exists to
        survive. With the floor in place the first block still goes through
        (1 in an assumed 20 is exactly the 5% cap), so genuine fraud is never
        ignored at startup, but the second in a row is already held.
        """
        with self.lock:
            if not wants_block:
                self.recent.append(0)
                return True, None

            denominator = max(len(self.recent) + 1, self.min_sample)
            prospective = (sum(self.recent) + 1) / denominator
            if prospective > self.max_rate:
                self.recent.append(0)  # downgraded: not counted as a block
                return False, (
                    f"block-rate gate: allowing this block would put the rolling "
                    f"block rate at {prospective:.1%} over the last {denominator} "
                    f"decisions, above the {self.max_rate:.1%} cap. "
                    f"Downgraded to review."
                )
            self.recent.append(1)
            return True, None

    def snapshot(self) -> dict:
        with self.lock:
            enforced = (sum(self.recent) / max(len(self.recent), self.min_sample)
                        if self.recent else 0.0)
            observed = (sum(self.recent) / len(self.recent)) if self.recent else 0.0
            return {"window_size": len(self.recent), "max_window": self.window,
                    "block_rate": enforced, "observed_rate": observed,
                    "blocks_in_window": int(sum(self.recent)),
                    "max_block_rate": self.max_rate, "min_sample": self.min_sample,
                    "warming_up": len(self.recent) < self.min_sample,
                    "headroom": max(self.max_rate - enforced, 0.0)}

    def reset(self) -> None:
        with self.lock:
            self.recent.clear()


# -------------------------------------------------------------- review queue
class ReviewQueue:
    """Append-only human-review queue.

    Anything banded `review`, and anything the gate downgraded, lands here. A
    reviewer approves or declines it and that resolution is appended too, so the
    file is a replayable history rather than mutable state. Pending items are
    rebuilt from it on startup.

    We store what a human needs to judge the case -- score, amount, the ranked
    factors with their evidence -- and a hash of the input, never the raw record.
    """

    def __init__(self, path: Path = REVIEW_QUEUE):
        self.path = path
        self.lock = threading.Lock()
        self.pending: dict[str, dict] = {}
        self._replay()

    def _replay(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "queued":
                self.pending[e["request_id"]] = e
            elif e.get("event") == "resolved":
                self.pending.pop(e.get("request_id"), None)
        if self.pending:
            print(f"[api] review queue: {len(self.pending)} pending case(s) replayed")

    def _append(self, entry: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            print(json.dumps(entry), file=fh)

    def add(self, case: dict) -> None:
        entry = {"event": "queued", **case}
        with self.lock:
            self.pending[case["request_id"]] = entry
            self._append(entry)

    def resolve(self, request_id: str, action: str, reviewer: str,
                note: str = "") -> dict:
        with self.lock:
            case = self.pending.pop(request_id, None)
            if case is None:
                raise KeyError(request_id)
            entry = {"event": "resolved", "request_id": request_id,
                     "ts": datetime.now(timezone.utc).isoformat(),
                     "action": action, "reviewer": reviewer, "note": note,
                     "risk_score": case.get("risk_score"),
                     "queued_reason": case.get("queued_reason")}
            self._append(entry)
            return entry

    def list_pending(self, n: int = 100) -> list[dict]:
        with self.lock:
            items = sorted(self.pending.values(),
                           key=lambda e: e.get("risk_score", 0.0), reverse=True)
            return items[:max(n, 1)]

    def stats(self) -> dict:
        resolved = approved = declined = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event") == "resolved":
                    resolved += 1
                    approved += e.get("action") == "approve"
                    declined += e.get("action") == "decline"
        with self.lock:
            pend = len(self.pending)
        return {"pending": pend, "resolved": resolved,
                "approved": int(approved), "declined": int(declined)}


# ------------------------------------------------------------------- service
class RiskService:
    def __init__(self):
        cfg = self._serving_config()
        self.config = cfg
        self.model_version = cfg.get("model_version", DEFAULT_MODEL_VERSION)
        self.shipped = cfg.get("shipped_model", "catboost")
        self.weights = cfg.get("ensemble_weights", {}) or {}

        self.members = self._load_members()
        if not self.members:
            raise RuntimeError("no model artefacts found -- run `python run.py train`")

        self.block_threshold = float(cfg.get("block_threshold", 0.5))
        self.review_threshold = float(cfg.get("review_threshold",
                                              self.block_threshold * 0.5))
        self.calibrator = Calibration.load()
        self.context = FeatureContext.load()
        self.featurizer = ServingFeaturizer()
        self.explainer_name = self._pick_explainer()

        self.audit_lock = threading.Lock()
        self.gate = BlockRateGate()
        self.review_queue = ReviewQueue()

        print(f"[api] ready. serving={self.shipped} explained_by={self.explainer_name} "
              f"block>={self.block_threshold:.4f} review>={self.review_threshold:.4f} "
              f"calibrator={self.calibrator.method} "
              f"gate<={MAX_BLOCK_RATE:.0%} over {ROLLING_WINDOW}")

    # ------------------------------------------------------------- loading
    @staticmethod
    def _serving_config() -> dict:
        if SERVING_CONFIG.exists():
            return json.loads(SERVING_CONFIG.read_text(encoding="utf-8"))
        print("[api] WARNING: serving_config.json missing (run `python run.py assess`); "
              "falling back to threshold 0.50")
        return {}

    def _load_members(self) -> dict:
        wanted = list(self.weights) if self.shipped == "ensemble" else [self.shipped]
        members = {}
        for name in wanted:
            try:
                members[name] = load(name)
            except Exception as exc:  # noqa: BLE001 - a missing member is survivable
                print(f"[api] WARNING: could not load {name}: {exc}")
        return members

    def _pick_explainer(self) -> str:
        """TreeSHAP-capable member, preferring the heaviest-weighted one."""
        capable = [n for n in self.members if n in ("xgboost", "catboost")]
        if not capable:
            return "none"
        if self.shipped in capable:
            return self.shipped
        return max(capable, key=lambda n: self.weights.get(n, 0.0))

    # ------------------------------------------------------------- scoring
    def _raw_score(self, X: pd.DataFrame) -> float:
        preds = {n: predict(n, m, X) for n, m in self.members.items()}
        if self.shipped == "ensemble" and len(preds) > 1:
            return float(ensemble_predict(preds, self.weights)[0])
        return float(preds[self.shipped][0]) if self.shipped in preds \
            else float(next(iter(preds.values()))[0])

    def _shap(self, X: pd.DataFrame) -> np.ndarray | None:
        name = self.explainer_name
        if name == "none":
            return None
        model = self.members[name]
        try:
            if name == "xgboost":
                import xgboost as xgb

                d = xgb.DMatrix(X, enable_categorical=True)
                return model.predict(d, pred_contribs=True)[0, :-1]
            from catboost import Pool

            from src.data import catboost_frame, categorical_indices

            pool = Pool(catboost_frame(X), cat_features=categorical_indices(X))
            return model.get_feature_importance(pool, type="ShapValues")[0, :-1]
        except Exception as exc:  # noqa: BLE001 - explanations must never 500
            print(f"[api] WARNING: SHAP failed ({exc}); returning no factors")
            return None

    def top_factors(self, X: pd.DataFrame, k: int = 4) -> list[dict]:
        contribs = self._shap(X)
        if contribs is None:
            return []
        cols = list(X.columns)
        order = np.argsort(np.abs(contribs))[::-1][:k]
        out = []
        for i in order:
            col = cols[i]
            value = X[col].iloc[0]
            if col in CATEGORICAL:
                value = None if pd.isna(value) else str(value)
            else:
                value = None if pd.isna(value) else float(value)
            factor = {"feature": col, "label": label(col),
                      "shap_value": round(float(contribs[i]), 6),
                      "direction": "increases risk" if contribs[i] > 0
                                   else "decreases risk"}
            factor.update(self.context.describe(col, value))
            self._note_disagreement(factor)
            out.append(factor)
        return out

    @staticmethod
    def _note_disagreement(factor: dict) -> None:
        """Flag when the local attribution and the group average point apart.

        A SHAP value is a LOCAL attribution for one transaction; a base rate is
        a MARGINAL average over a group. They can legitimately disagree -- chip
        transactions are low-risk on average, yet chip can still raise the score
        for a particular combination of amount, category and time.

        Left unexplained this reads as a contradiction ("increases risk" beside
        "0.06% of these were fraud") and a reviewer stops trusting the whole
        panel. So we say which is which rather than hiding one of them.
        """
        lift = factor.get("lift")
        if lift is None:
            return
        raises = factor.get("direction") == "increases risk"
        if raises and lift < 1.0:
            factor["evidence"] += (" — note: this group is not unusually risky "
                                   "on its own; the model raised the score on "
                                   "how it combines with the other details")
        elif not raises and lift > 1.5:
            factor["evidence"] += (" — note: this group is risky on average, "
                                   "but for this transaction the model treated "
                                   "it as reassuring")

    def band(self, score: float) -> tuple[Decision, str]:
        if score >= self.block_threshold:
            return "block", "RULE_BLOCK_ABOVE_COST_OPTIMAL_THRESHOLD"
        if score >= self.review_threshold:
            return "review", "RULE_REVIEW_BAND_BELOW_BLOCK_THRESHOLD"
        return "allow", "RULE_ALLOW_BELOW_REVIEW_BAND"

    def score(self, txn: Transaction) -> ScoreResponse:
        t0 = time.perf_counter()
        raw = txn.model_dump()
        X, missing = self.featurizer.build(raw)

        uncalibrated = self._raw_score(X)
        score = float(self.calibrator.predict(np.array([uncalibrated]))[0])
        factors = self.top_factors(X)

        raw_decision, rule_id = self.band(score)
        allowed, gate_reason = self.gate.admit(raw_decision == "block")
        decision: Decision = raw_decision
        if raw_decision == "block" and not allowed:
            decision, rule_id = "review", "RULE_GATE_DOWNGRADE_BLOCK_TO_REVIEW"

        resp = ScoreResponse(
            request_id=str(uuid.uuid4()),
            risk_score=round(score, 6),
            risk_percent=f"{score * 100:.2f}%",
            decision=decision,
            raw_decision=raw_decision,
            gated=decision != raw_decision,
            gate_reason=gate_reason,
            top_factors=factors,
            rule_id=rule_id,
            thresholds={"review": self.review_threshold,
                        "block": self.block_threshold},
            model_version=self.model_version,
            scored_by=self.shipped,
            explained_by=self.explainer_name,
            missing_fields=missing,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            base_rate=self.context.band_fraud_rate(score),
        )
        self._audit(resp, raw)
        if resp.decision == "review":
            self._queue_for_review(resp, raw)
        return resp

    # -------------------------------------------------------------- logging
    def _input_hash(self, raw: dict) -> str:
        payload = "|".join(f"{k}={raw.get(k)}" for k in sorted(raw))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _audit(self, resp: ScoreResponse, raw: dict) -> None:
        """Append-only. We log a hash of the input, never the input itself."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": resp.request_id,
            "input_sha256": self._input_hash(raw),
            "amount": float(raw.get("amount", 0.0) or 0.0),
            "risk_score": resp.risk_score,
            "decision": resp.decision,
            "raw_decision": resp.raw_decision,
            "gated": resp.gated,
            "gate_reason": resp.gate_reason,
            "rule_id": resp.rule_id,
            "threshold_used": resp.thresholds["block"],
            "review_threshold": resp.thresholds["review"],
            "model_version": resp.model_version,
            "scored_by": resp.scored_by,
            "calibrator": self.calibrator.method,
            "top_factors": [f["feature"] for f in resp.top_factors],
            "missing_fields": resp.missing_fields,
            "rolling_block_rate": round(self.gate.enforced_rate(), 4),
            "latency_ms": resp.latency_ms,
        }
        with self.audit_lock:
            with open(AUDIT_LOG, "a", encoding="utf-8") as fh:
                print(json.dumps(entry), file=fh)

    def _queue_for_review(self, resp: ScoreResponse, raw: dict) -> None:
        self.review_queue.add({
            "request_id": resp.request_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "input_sha256": self._input_hash(raw),
            "risk_score": resp.risk_score,
            "risk_percent": resp.risk_percent,
            "amount": float(raw.get("amount", 0.0) or 0.0),
            "merchant_category": raw.get("mcc_category"),
            "queued_reason": ("gate downgrade" if resp.gated
                              else "score in review band"),
            "gate_reason": resp.gate_reason,
            "rule_id": resp.rule_id,
            "thresholds": resp.thresholds,
            "model_version": resp.model_version,
            "top_factors": resp.top_factors,
            "base_rate": resp.base_rate,
            "missing_fields": resp.missing_fields,
        })


# ----------------------------------------------------------------- app wiring
app = FastAPI(
    title="Fraud Risk Manager",
    version=DEFAULT_MODEL_VERSION,
    description="Defense-only merchant transaction risk scoring with bounded, "
                "gated decisioning, human review and an append-only audit trail.",
)
_service: RiskService | None = None


@app.exception_handler(RequestValidationError)
def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a clean 422 for malformed input.

    FastAPI's default handler echoes the offending value back. When that value
    is a non-finite float -- {"amount": NaN}, which several JSON parsers accept
    -- serialising the response itself raises, so a bad request crashes the
    response path instead of being rejected. Non-finite values are stripped.
    """

    def clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return f"<non-finite: {v}>"
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    errors = [{"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg", ""),
               "type": e.get("type", ""), "input": clean(e.get("input"))}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(json.JSONDecodeError)
def _bad_json_handler(request: Request, exc: json.JSONDecodeError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": f"malformed JSON: {exc}"})


def get_service() -> RiskService:
    global _service
    if _service is None:
        _service = RiskService()
    return _service


@app.on_event("startup")
def _startup() -> None:
    if os.environ.get("FRM_LAZY_LOAD") != "1":
        get_service()


@app.get("/health")
def health() -> dict:
    s = get_service()
    return {"status": "ok", "model_version": s.model_version,
            "scored_by": s.shipped, "explained_by": s.explainer_name,
            "ensemble_weights": s.weights,
            "thresholds": {"review": s.review_threshold, "block": s.block_threshold},
            "calibrator": s.calibrator.method,
            "gate": s.gate.snapshot(),
            "review": s.review_queue.stats(),
            "context": {"available": s.context.available,
                        "reference_rows": s.context.n_reference,
                        "overall_fraud_rate": s.context.overall_rate}}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    try:
        return get_service().score(txn)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"scoring failed: {exc}") from exc


@app.get("/gate")
def gate_status() -> dict:
    return get_service().gate.snapshot()


@app.post("/gate/reset")
def gate_reset() -> dict:
    """Demo convenience: clear the rolling window between takes."""
    get_service().gate.reset()
    return {"status": "gate window cleared"}


@app.get("/review/queue")
def review_queue(n: int = 50) -> dict:
    svc = get_service()
    return {"pending": svc.review_queue.list_pending(n),
            "stats": svc.review_queue.stats(),
            "thresholds": {"review": svc.review_threshold,
                           "block": svc.block_threshold}}


@app.post("/review/resolve")
def review_resolve(res: ReviewResolution) -> dict:
    """A human approves or declines a queued case.

    The resolution is appended to the review log AND to the audit trail, so the
    final disposition of every gated decision is recoverable.
    """
    svc = get_service()
    try:
        entry = svc.review_queue.resolve(res.request_id, res.action,
                                         res.reviewer, res.note)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"no pending review {res.request_id}") from None
    with svc.audit_lock:
        with open(AUDIT_LOG, "a", encoding="utf-8") as fh:
            print(json.dumps({
                "ts": entry["ts"], "request_id": res.request_id,
                "decision": "human_review_resolved", "review_action": res.action,
                "reviewer": res.reviewer, "note": res.note,
                "risk_score": entry.get("risk_score"),
                "rule_id": "RULE_HUMAN_REVIEW_RESOLUTION",
                "model_version": svc.model_version,
            }), file=fh)
    return {"status": "resolved", **entry}


@app.get("/review/stats")
def review_stats() -> dict:
    return get_service().review_queue.stats()


@app.get("/audit")
def audit(n: int = 50) -> dict:
    if not AUDIT_LOG.exists():
        return {"entries": [], "total": 0}
    lines = AUDIT_LOG.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[-max(n, 1):]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"entries": entries, "total": len(lines),
            "gate": get_service().gate.snapshot()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
