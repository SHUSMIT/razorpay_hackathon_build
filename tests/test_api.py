"""API robustness and, most importantly, that the bounded-decisioning gate
actually fires under a burst instead of blocking the whole shop."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.config import BASELINE_MODEL, TUNED_MODEL


@pytest.fixture(scope="module")
def client():
    if not (TUNED_MODEL.exists() or BASELINE_MODEL.exists()):
        pytest.skip("no model artefact -- run `make train`")
    from src.api import app, get_service

    get_service().gate.reset()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_gate(client):
    from src.api import get_service

    get_service().gate.reset()


VALID = {"Time": 40000.0, "Amount": 149.62}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert 0 < body["thresholds"]["review"] <= body["thresholds"]["block"] < 1


def test_scores_a_valid_transaction(client):
    body = client.post("/score", json=VALID).json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"allow", "review", "block"}
    assert len(body["top_factors"]) == 3
    assert body["rule_id"].startswith("RULE_")
    assert body["latency_ms"] >= 0


def test_missing_optional_fields_are_imputed_not_fatal(client):
    body = client.post("/score", json={"Amount": 12.5}).json()
    assert len(body["imputed_features"]) == 28, "V1..V28 should be flagged as imputed"
    assert body["decision"] in {"allow", "review", "block"}


def test_missing_required_field_is_rejected(client):
    assert client.post("/score", json={"Time": 1.0}).status_code == 422


@pytest.mark.parametrize("payload", [
    {"Amount": "not-a-number", "Time": 0},
    {"Amount": -5.0},
    {"Amount": 1e12},
    {"Amount": 100.0, "V1": "banana"},
    {"Amount": None},
    {"Amount": [1, 2, 3]},
])
def test_malformed_requests_are_rejected_without_crashing(client, payload):
    r = client.post("/score", json=payload)
    assert r.status_code == 422, f"{payload} -> {r.status_code}"
    assert client.get("/health").json()["status"] == "ok", "service survived"


@pytest.mark.parametrize("body", [
    '{"Amount": NaN}',        # not legal JSON, but many parsers accept it
    '{"Amount": Infinity}',
    '{"Amount": 1.0,',        # truncated body
    "not json at all",
])
def test_non_finite_and_broken_bodies_are_rejected(client, body):
    """Sent as a raw body: `float("nan")` cannot even be serialised by a
    conforming JSON client, so the only way this reaches a server is as a raw
    non-standard token."""
    r = client.post("/score", content=body,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422, f"{body!r} -> {r.status_code}"
    assert client.get("/health").json()["status"] == "ok", "service survived"


def test_extreme_but_valid_amount_is_handled(client):
    r = client.post("/score", json={"Time": 0.0, "Amount": 999_999_999.0})
    assert r.status_code == 200
    assert 0.0 <= r.json()["risk_score"] <= 1.0


# ------------------------------------------------------------ bounded gating
RAW_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@pytest.fixture(scope="module")
def high_risk() -> dict:
    """A real held-out fraud transaction the model scores above the block
    threshold. Using real data rather than a hand-made vector means this test
    exercises the gate the way production traffic would."""
    from src.api import get_service
    from src.evaluate import predict
    from src.train import xy

    svc = get_service()
    X, y = xy("test")
    frauds = X[y == 1]
    scores = predict(svc.booster, frauds)
    hits = frauds[scores >= svc.block_threshold]
    if hits.empty:
        pytest.skip("no held-out fraud scores above the block threshold")
    row = hits.iloc[0]
    # Reconstruct the raw request shape from the engineered row.
    payload = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
    payload["Amount"] = float(row["Amount"])
    payload["Time"] = float(row["hour_of_day"]) * 3600.0
    return payload


def test_gate_downgrades_blocks_under_a_burst(client, high_risk):
    HIGH_RISK = high_risk
    from src.api import get_service

    svc = get_service()
    results = [client.post("/score", json=HIGH_RISK).json() for _ in range(100)]
    assert all(r["raw_decision"] == "block" for r in results), \
        "fixture is not high-risk enough to exercise the gate"

    served_blocks = sum(r["decision"] == "block" for r in results)
    downgraded = [r for r in results if r["gated"]]

    assert downgraded, "gate never fired under a 100-request block burst"
    assert all(r["decision"] == "review" for r in downgraded)
    assert all(r["rule_id"] == "RULE_GATE_DOWNGRADE_BLOCK_TO_REVIEW" for r in downgraded)
    assert all("block-rate gate" in r["gate_reason"] for r in downgraded)
    # The whole point: the served block rate stays under the cap.
    assert served_blocks / len(results) <= svc.gate.max_rate + 0.01
    assert svc.gate.current_rate() <= svc.gate.max_rate + 1e-9


def test_gate_does_not_interfere_with_low_risk_traffic(client):
    results = [client.post("/score", json=VALID).json() for _ in range(30)]
    assert not any(r["gated"] for r in results)
    assert client.get("/gate").json()["block_rate"] == 0.0


def test_gate_lets_early_blocks_through_before_min_sample(client, high_risk):
    """Gating must not kick in on the very first request of a cold service --
    otherwise the bound would suppress genuine fraud blocking at startup."""
    first = client.post("/score", json=high_risk).json()
    assert first["decision"] == "block" and not first["gated"]


# ----------------------------------------------------------------- audit log
def test_every_decision_is_audited(client):
    from src.config import AUDIT_LOG

    before = len(AUDIT_LOG.read_text().splitlines()) if AUDIT_LOG.exists() else 0
    for _ in range(5):
        client.post("/score", json=VALID)
    after = AUDIT_LOG.read_text().splitlines()
    assert len(after) == before + 5, "audit log is not append-only per decision"

    entry = json.loads(after[-1])
    for field in ("ts", "input_sha256", "risk_score", "decision", "raw_decision",
                  "rule_id", "threshold_used", "model_version", "gated"):
        assert field in entry, f"audit entry missing {field}"
    assert len(entry["input_sha256"]) == 64
    assert "amount" in entry, "amount should be logged for review triage"
    assert not any(k.startswith("V") for k in entry), \
        "raw PCA feature vector must not be written to the audit log"


def test_audit_endpoint_returns_recent_entries(client):
    client.post("/score", json=VALID)
    body = client.get("/audit", params={"n": 3}).json()
    assert len(body["entries"]) <= 3 and body["total"] >= 1
    assert "gate" in body
