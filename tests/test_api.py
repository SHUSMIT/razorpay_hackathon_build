"""API robustness, and the parts that make this more than a model behind HTTP:
the block-rate gate, the human review queue and the audit trail.

These need trained artefacts, so they skip cleanly on a fresh checkout rather
than failing and hiding real breakage.
"""
from __future__ import annotations

import json

import pytest

from src.config import SERVING_CONFIG
from src.models import available

# The service needs BOTH trained models and a serving config (thresholds are
# written by assess.py). Mid-pipeline the models exist but the config does not,
# so check for both -- otherwise these error instead of skipping, which hides
# whether anything is genuinely broken.
_READY = bool(available()) and SERVING_CONFIG.exists()
pytestmark = pytest.mark.skipif(
    not _READY,
    reason="pipeline not complete yet -- run `python run.py all`")

FULL = {"amount": 142.50, "date": "2019-06-14T02:31:00",
        "use_chip": "Online Transaction", "mcc": 5812, "credit_limit": 12000.0,
        "card_brand": "Visa", "card_type": "Credit", "has_chip": True,
        "errors": None}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import src.api as api

    with TestClient(api.app) as c:
        yield c


# ------------------------------------------------------------------- basics
def test_health_reports_a_coherent_operating_point(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    t = body["thresholds"]
    assert 0 < t["review"] <= t["block"] < 1, \
        "the review band must sit below the block threshold"


def test_score_returns_a_bounded_decision_and_a_percentage(client):
    body = client.post("/score", json=FULL).json()
    assert body["decision"] in {"allow", "review", "hold", "block"}
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_percent"].endswith("%")


def test_amount_only_request_still_scores(client):
    """Real traffic arrives incomplete; the service must not require the world."""
    body = client.post("/score", json={"amount": 55.0}).json()
    assert body["decision"] in {"allow", "review", "hold", "block"}
    assert body["missing_fields"], "gaps should be declared, not hidden"


@pytest.mark.parametrize("payload", [
    {}, {"amount": "abc"}, {"amount": None}, {"amount": float("inf")},
    {"amount": 1e12}, {"amount": 10, "mcc": "not-a-code"},
])
def test_malformed_requests_are_rejected_not_crashed(client, payload):
    r = client.post("/score", data=json.dumps(payload),
                    headers={"Content-Type": "application/json"})
    assert r.status_code in (422, 400), f"{payload} produced {r.status_code}"


def test_explanations_are_human_readable(client):
    """A reviewer must never be handed a bare column name."""
    body = client.post("/score", json=FULL).json()
    for f in body["top_factors"]:
        assert f["label"] != f["feature"], "factor was not given a readable label"
        assert f["evidence"], "factor carries no evidence for a human"
        assert f["direction"] in ("increases risk", "decreases risk")


# ------------------------------------------------------------- review queue
def test_review_cases_reach_a_human_and_can_be_resolved(client):
    high = {**FULL, "amount": 2000.0, "errors": "Bad CVV"}
    for _ in range(120):
        client.post("/score", json=high)

    q = client.get("/review/queue", params={"n": 50}).json()
    if not q["pending"]:
        pytest.skip("no case landed in review for this model/threshold")

    case = q["pending"][0]
    assert case["top_factors"], "a reviewer needs the evidence, not just a score"
    assert case["queued_reason"] in {"score in review band", "legacy gate review"}

    before = client.get("/review/stats").json()
    r = client.post("/review/resolve", json={
        "request_id": case["request_id"], "action": "decline",
        "reviewer": "pytest", "note": "confirmed"}).json()
    assert r["status"] == "resolved"
    after = client.get("/review/stats").json()
    assert after["declined"] == before["declined"] + 1
    assert after["pending"] == before["pending"] - 1


def test_resolving_an_unknown_case_is_a_404(client):
    r = client.post("/review/resolve", json={
        "request_id": "does-not-exist", "action": "approve"})
    assert r.status_code == 404


def test_review_resolution_is_rejected_for_an_invalid_action(client):
    r = client.post("/review/resolve", json={
        "request_id": "x", "action": "delete-everything"})
    assert r.status_code == 422


# --------------------------------------------------------------- audit trail
def test_every_decision_is_audited_without_storing_raw_input(client):
    client.post("/score", json=FULL)
    body = client.get("/audit", params={"n": 5}).json()
    assert body["entries"]
    entry = body["entries"][-1]
    for field in ("ts", "request_id", "input_sha256", "risk_score", "decision",
                  "rule_id", "threshold_used", "model_version", "scored_by"):
        assert field in entry, f"audit entry is missing {field}"
    assert len(entry["input_sha256"]) == 64
    assert "card_brand" not in entry, "raw transaction detail leaked into the log"
