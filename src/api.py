"""FastAPI risk-scoring service.

Defense-only. The service scores a transaction and returns a bounded decision:

  allow / review / block

Four things make this more than a model behind HTTP:

  1. THRESHOLDS COME FROM THE COST CURVE, not from 0.5. The block threshold is
     whatever minimised money lost, selected on a slice of validation that the
     models, the blend weights and the calibrator never touched. The review
     band beneath it is sized by how much traffic a human team can actually
     look at, not by a magic constant.
  2. HUMAN REVIEW IS CONFIDENCE-BASED. Only transactions in the calibrated
     review band are sent to a reviewer; confident blocks are actioned
     automatically, even during a burst.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.calibration import Calibration
from src.config import (
    AUDIT_LOG,
    MODELS,
    REVIEW_QUEUE,
    SERVING_CONFIG,
)
from src.context import FeatureContext
from src.models import ensemble_predict, load, predict
from src.schema import CATEGORICAL, FEATURES, label
from src.serve_features import ServingFeaturizer


def _load_local_env() -> None:
    """Load this project's local key without overwriting real process settings."""
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("GROQ_API_KEY=") and not os.environ.get("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
            return


_load_local_env()

Decision = Literal["allow", "review", "hold", "block"]
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
    reviewer_summary: str | None = None
    narrative_provider: str | None = None


class ReviewResolution(BaseModel):
    request_id: str
    action: Literal["approve", "decline"]
    reviewer: str = Field("demo-reviewer", max_length=64)
    note: str = Field("", max_length=500)


class GroqNarrator:
    """Optional, grounded reviewer narrative. It never changes a decision."""

    models = ("openai/gpt-oss-20b",)
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def explain(self, raw: dict, score: float, decision: str,
                factors: list[dict], base_rate: dict | None) -> tuple[str | None, str | None]:
        if not self.enabled:
            return None, None
        evidence = [{
            "signal": item.get("label"), "direction": item.get("direction"),
            "evidence": item.get("evidence"),
        } for item in factors]
        model_result = {"decision": decision}
        if decision != "review":
            model_result["risk_score"] = round(score, 4)
        prompt = {
            "transaction": raw,
            "model_result": model_result,
            "measured_evidence": evidence,
            "score_band_history": base_rate,
        }
        system = (
            "You are a payment-risk reviewer. Write a clear, grounded explanation "
            "for an operator in 3 short sentences (at most 110 words). Sentence 1 "
            "must state the supplied prediction only. Sentence 2 must name the two "
            "strongest risk-increasing facts and explain their combined relevance. "
            "Sentence 3 must name any supplied counter-evidence and say what a human "
            "should verify. Use only the transaction and measured evidence provided; "
            "never invent facts, claim certainty, give fraud-evasion advice, or alter "
            "the decision. If the decision is review, say it was flagged for human "
            "review and never state a numeric score, confidence, or recommendation."
        )
        for model in self.models:
            try:
                response = requests.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": model,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": json.dumps(prompt, default=str)}],
                          "temperature": 0.2, "reasoning_effort": "low",
                          "max_completion_tokens": 500},
                    timeout=20,
                )
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                text = (message.get("content") or "").strip()
                if text:
                    return text[:1_200], model
                print(f"[api] LLM narrative empty: {model}")
            except requests.HTTPError:
                print(f"[api] LLM narrative failed: {model} HTTP {response.status_code}")
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                print(f"[api] LLM narrative failed: {model} {type(exc).__name__}")
        return None, None


# -------------------------------------------------------------- review queue
class ReviewQueue:
    """Append-only human-review queue.

    Both score-band reviews and gate-overflow reviews land here. There must not
    be a second, invisible "verification" register when a human has the final
    disposition. A review resolution is appended, and pending items are rebuilt
    on startup.

    We store what a human needs to judge the case -- score, amount, the ranked
    factors with their evidence -- and a hash of the input, never the raw record.
    """

    def __init__(self, path: Path = REVIEW_QUEUE):
        self.path = path
        self.lock = threading.Lock()
        self.pending: dict[str, dict] = {}
        self._replay()

    def _replay(self) -> None:
        if self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines()
        else:
            lines = []
        for line in lines:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "queued":
                if e.get("queued_reason") == "gate downgrade":
                    e["queued_reason"] = "legacy gate review"
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
        self.narrator = GroqNarrator()

        self.audit_lock = threading.Lock()
        self.review_queue = ReviewQueue()

        print(f"[api] ready. serving={self.shipped} explained_by={self.explainer_name} "
              f"block>={self.block_threshold:.4f} review>={self.review_threshold:.4f} "
              f"calibrator={self.calibrator.method}")

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

    def top_factors(self, X: pd.DataFrame, k: int = 6) -> list[dict]:
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

        decision, rule_id = self.band(score)
        raw_decision = decision

        base_rate = self.context.band_fraud_rate(score)
        summary, narrative_model = self.narrator.explain(
            raw, score, decision, factors, base_rate)

        resp = ScoreResponse(
            request_id=str(uuid.uuid4()),
            risk_score=round(score, 6),
            risk_percent=f"{score * 100:.2f}%",
            decision=decision,
            raw_decision=raw_decision,
            gated=False,
            gate_reason=None,
            top_factors=factors,
            rule_id=rule_id,
            thresholds={"review": self.review_threshold,
                        "block": self.block_threshold},
            model_version=self.model_version,
            scored_by=self.shipped,
            explained_by=self.explainer_name,
            missing_fields=missing,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            base_rate=base_rate,
            reviewer_summary=summary,
            narrative_provider=(f"Groq / {narrative_model}" if narrative_model else None),
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
            "queued_reason": "score in review band",
            "gate_reason": resp.gate_reason,
            "rule_id": resp.rule_id,
            "thresholds": resp.thresholds,
            "model_version": resp.model_version,
            "top_factors": resp.top_factors,
            "base_rate": resp.base_rate,
            "reviewer_summary": resp.reviewer_summary,
            "narrative_provider": resp.narrative_provider,
            "missing_fields": resp.missing_fields,
        })

# ----------------------------------------------------------------- app wiring
app = FastAPI(
    title="Fraud Risk Manager",
    version=DEFAULT_MODEL_VERSION,
    description="Defense-only merchant transaction risk scoring with bounded, "
                "confidence-based human review and an append-only audit trail.",
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
            "review": s.review_queue.stats(),
            "narrative": {"configured": s.narrator.enabled,
                          "models": list(s.narrator.models)},
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

    The resolution is appended to the review log AND to the audit trail.
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
    return {"entries": entries, "total": len(lines)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
