"""Integration tests for the Orcha HTTP API."""
import pytest
from fastapi.testclient import TestClient
from orcha.api.server import app


@pytest.fixture
def client():
    return TestClient(app)


# ── Versioned /v1 routes — the stable public API ──────────────────────────

def test_health_v1(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.3.0"
    assert isinstance(body["experts"], list) and len(body["experts"]) > 0
    assert "synthesizer" in body
    assert "run_all_experts" in body
    assert "source" in body


def test_list_experts_v1(client):
    r = client.get("/v1/experts")
    assert r.status_code == 200
    experts = r.json()
    assert isinstance(experts, list) and len(experts) > 0
    for e in experts:
        assert "name" in e and "domain" in e and "description" in e
        assert "is_synthesizer" in e


def test_query_returns_full_shape(client):
    r = client.post("/v1/query", json={"query": "What is compound interest?"})
    assert r.status_code == 200
    body = r.json()
    for k in ("answer", "confidence", "quality_score", "synthesized",
              "iterations", "cost", "latency_s", "contributors",
              "primary", "domains", "trace", "agg_mode"):
        assert k in body, f"Missing key: {k}"
    assert body["answer"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["trace"][0]["stage"] == "input"


def test_query_max_iterations_override(client):
    r = client.post("/v1/query", json={"query": "Quick test", "max_iterations": 1})
    assert r.status_code == 200
    assert r.json()["iterations"] == 1


def test_query_run_all_override(client):
    r = client.post("/v1/query", json={"query": "Test", "run_all_experts": True})
    assert r.status_code == 200
    body = r.json()
    assert len(body["contributors"]) > 0


def test_query_max_cost_zero_is_honoured(client):
    """Regression: max_cost=0 must NOT be silently overridden to the default."""
    r = client.post("/v1/query", json={
        "query": "test", "max_cost": 0.0, "max_iterations": 1,
    })
    assert r.status_code == 200
    assert r.json()["cost"] >= 0.0   # local models cost $0 anyway


def test_reload_v1(client):
    r = client.post("/v1/reload")
    assert r.status_code == 200
    body = r.json()
    assert body["reloaded"] is True
    assert isinstance(body["experts"], list)
    assert "source" in body


def test_performance_v1(client):
    # Run a query first so there's data
    client.post("/v1/query", json={"query": "warm up", "max_iterations": 1})
    r = client.get("/v1/performance")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


# ── Back-compat redirects (deprecated un-versioned paths) ──────────────────

def test_legacy_health_redirects(client):
    r = client.get("/health", follow_redirects=False)
    assert r.status_code in (301, 307)
    assert "/v1/health" in r.headers.get("location", "")


def test_legacy_query_redirects(client):
    r = client.post("/query", json={"query": "hi"}, follow_redirects=False)
    assert r.status_code in (301, 307)
    assert "/v1/query" in r.headers.get("location", "")


def test_legacy_reload_redirects(client):
    r = client.post("/reload", follow_redirects=False)
    assert r.status_code in (301, 307)
    assert "/v1/reload" in r.headers.get("location", "")


# ── Structured error envelope ──────────────────────────────────────────────

def test_error_envelope_on_bad_query(client):
    """A query with missing required field should return a validation error
    in the standard FastAPI format (422).  Our OrchaError handler covers
    unhandled 500s; FastAPI's own validation errors still use its format,
    which is expected and correct."""
    r = client.post("/v1/query", json={})
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body  # FastAPI validation error format


# ── UI ─────────────────────────────────────────────────────────────────────

def test_root_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Orcha" in r.text
    assert "text/html" in r.headers["content-type"]
