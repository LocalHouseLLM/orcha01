"""
orcha.api.server
================
FastAPI server that exposes Orcha over HTTP and serves the web UI.

The orchestrator is built inside a FastAPI ``lifespan`` handler, NOT at
module import time. Importing this module (for ``TestClient``, tooling,
or programmatic use) is therefore instant and never blocks on a network
probe or triggers an event-loop conflict.

Versioning
----------
The stable public API lives under ``/v1`` (e.g. ``POST /v1/query``).
The un-versioned paths (``/query``, ``/health``, …) are retained as
thin redirects for backwards compatibility but are considered
deprecated and will be removed in a future major release.

Errors
------
All error responses share one envelope::

    {"error": {"type": "...", "message": "...", "status": 500, "trace_id": "..."}}

so clients can parse failures uniformly. Internal tracebacks are never
leaked to the client.

Start:
    uvicorn orcha.api.server:app --reload --port 8420

Then open http://localhost:8420 for the dashboard.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..experts.mock import load_mock_experts
from ..experts.ollama import list_ollama_models
from ..experts.registry import LocalModelRegistry
from ..observability import configure_logging, get_logger
from ..orchestrator import Orchestrator
from ..settings import settings

API_VERSION = "0.3.0"
_log = get_logger("orcha.api")


# ── Orchestrator builder (single source of truth) ─────────────────────────────

async def _build_registry_from_ollama() -> Optional[LocalModelRegistry]:
    """
    Probe the local Ollama daemon and register every pulled model.

    Returns None if Ollama is not reachable or has no models — never raises.
    The caller then falls back to mock experts so the server always starts.

    This is an ``async`` function on purpose: it is always called from
    inside an event loop (the FastAPI lifespan or an async route handler),
    so it must ``await`` the discovery coroutine directly rather than call
    ``asyncio.run()``, which raises inside an already-running loop and
    would silently break discovery.
    """
    try:
        models = await list_ollama_models(settings.ollama_base_url)
    except Exception:
        return None
    if not models:
        return None

    registry = LocalModelRegistry()
    for m in models:
        if any(p in m.lower() for p in settings.ollama_skip_patterns):
            continue
        registry.add_ollama(m, base_url=settings.ollama_base_url)
    return registry


async def build_orchestrator() -> Orchestrator:
    """
    Try Ollama auto-discovery; fall back to mock experts so the server
    always starts cleanly even with no local models installed.

    This is the ONE place the startup/reload logic lives — ``/reload``
    delegates here too, so they can never drift apart.
    """
    registry = await _build_registry_from_ollama()
    if registry is not None and registry.names():
        experts     = registry.build()
        synthesizer = registry.pick_synthesizer()
        source      = "ollama"
        run_all     = settings.run_all_experts
    else:
        experts     = load_mock_experts()
        synthesizer = "mock_synthesizer"
        source      = "mock"
        run_all     = True   # mock mode is designed to use the full pool

    orc = Orchestrator(
        experts=experts,
        synthesizer_expert=synthesizer,
        run_all_experts=run_all,
        max_cost=settings.max_cost,
        max_latency_s=settings.max_latency_s,
        max_iterations=settings.max_iterations,
        use_embeddings=settings.use_embeddings,
    )
    orc._source = source  # type: ignore[attr-defined]  # informational only
    _log.info("orchestrator.built source=%s experts=%d synthesizer=%s",
              source, len(experts), synthesizer)
    return orc


# ── App state (built lazily at startup, not import) ───────────────────────────

class AppState:
    """Mutable holder for the live orchestrator. Cheap to swap on /reload."""
    orc: Optional[Orchestrator] = None


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the orchestrator when the server starts, not on import."""
    configure_logging()
    _state.orc = await build_orchestrator()
    try:
        yield
    finally:
        _state.orc = None


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Orcha API",
    version=API_VERSION,
    description=(
        "Local-first multi-model orchestration. Runs all your local LLMs "
        "in parallel and synthesizes one refined answer."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _orc() -> Orchestrator:
    """
    Return the live orchestrator. It is built during the lifespan startup
    event, so by the time any route runs it is guaranteed to be present.
    """
    if _state.orc is None:
        # Defensive: only reachable if a route somehow fires before startup
        # completed. Return a mock-backed orchestrator so the call never
        # hard-fails; the real one replaces it on the next startup.
        orc = Orchestrator(
            experts=load_mock_experts(),
            synthesizer_expert="mock_synthesizer",
            run_all_experts=True,
            max_cost=settings.max_cost,
            max_latency_s=settings.max_latency_s,
            max_iterations=settings.max_iterations,
        )
        orc._source = "mock"  # type: ignore[attr-defined]
        _state.orc = orc
    assert _state.orc is not None
    return _state.orc


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:           str = Field(..., min_length=1, max_length=20000)
    max_cost:        Optional[float] = Field(default=None, ge=0.0)
    max_iterations:  Optional[int]   = Field(default=None, ge=0)
    run_all_experts: Optional[bool]  = None


class QueryResponse(BaseModel):
    answer:        str
    confidence:    float
    quality_score: float
    synthesized:   bool
    agg_mode:      str
    iterations:    int
    cost:          float
    latency_s:     float
    contributors:  List[str]
    primary:       Optional[str]
    domains:       List[str]
    trace:         List[Dict[str, Any]]


class ExpertInfo(BaseModel):
    name:            str
    domain:          str
    description:     str
    version:         str
    is_synthesizer:  bool


class HealthResponse(BaseModel):
    status:          str
    version:         str
    source:          str
    synthesizer:     Optional[str]
    run_all_experts: bool
    experts:         List[str]


# ── Structured error envelope ─────────────────────────────────────────────────

class OrchaError(Exception):
    """Internal exception type carrying a normalized error payload."""
    def __init__(self, type_: str, message: str, status: int = 500):
        self.type = type_
        self.message = message
        self.status = status
        super().__init__(message)


@app.exception_handler(OrchaError)
async def _orcha_error_handler(request: Request, exc: OrchaError) -> JSONResponse:
    return _error_response(exc.type, exc.message, exc.status, request)


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the full traceback internally; never leak it to the client.
    _log.exception("unhandled_error path=%s", request.url.path)
    return _error_response(
        "internal_error",
        "An unexpected error occurred while processing the request.",
        500,
        request,
    )


def _error_response(
    type_: str, message: str, status: int, request: Request
) -> JSONResponse:
    envelope = {
        "error": {
            "type":    type_,
            "message": message,
            "status":  status,
            "trace_id": getattr(request.state, "trace_id", None),
        }
    }
    return JSONResponse(status_code=status, content=envelope)


# ── Versioned routes (/v1) — the stable public API ────────────────────────────

@app.get("/v1/health", response_model=HealthResponse)
async def health():
    orc = _orc()
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        source=getattr(orc, "_source", "unknown"),
        synthesizer=orc.synthesizer_expert,
        run_all_experts=orc.run_all_experts,
        experts=list(orc.experts.keys()),
    )


@app.get("/v1/experts", response_model=List[ExpertInfo])
async def list_experts():
    return [ExpertInfo(**e) for e in _orc().list_experts()]


@app.post("/v1/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    base = _orc()
    try:
        # Per-request overrides use explicit None checks — a client asking
        # for max_cost=0 (local-only, spend nothing) or max_iterations=0
        # must NOT be silently coerced to the settings default by a
        # truthy/falsy `or` fallback.
        if any(v is not None for v in
               (req.max_cost, req.max_iterations, req.run_all_experts)):
            orc = Orchestrator(
                experts=base.experts,
                synthesizer_expert=base.synthesizer_expert,
                run_all_experts=(
                    req.run_all_experts
                    if req.run_all_experts is not None
                    else base.run_all_experts
                ),
                max_cost=(
                    req.max_cost
                    if req.max_cost is not None
                    else settings.max_cost
                ),
                max_latency_s=settings.max_latency_s,
                max_iterations=(
                    req.max_iterations
                    if req.max_iterations is not None
                    else settings.max_iterations
                ),
            )
        else:
            orc = base

        result = await orc.run_async(req.query)
        return QueryResponse(**result.to_dict())
    except Exception as exc:
        _log.exception("query_failed query=%r", req.query[:80])
        raise OrchaError("query_failed", str(exc), 500)


@app.post("/v1/reload")
async def reload_experts():
    """Re-discover Ollama models. Call after `ollama pull <model>`."""
    _state.orc = await build_orchestrator()
    orc = _state.orc
    assert orc is not None
    return {
        "reloaded": True,
        "source":   getattr(orc, "_source", "unknown"),
        "experts":  list(orc.experts.keys()),
    }


@app.get("/v1/performance")
async def performance():
    """Per-expert selector performance history."""
    return _orc().selector_performance()


# ── Deprecated un-versioned routes → redirect to /v1 (back-compat) ────────────

_DEPRECATED = {
    "/health":      "/v1/health",
    "/experts":     "/v1/experts",
    "/performance": "/v1/performance",
}
# POST routes get a 307 so the method/body are preserved; GET get a 301.
for _old, _new in _DEPRECATED.items():
    @app.get(_old, include_in_schema=False)
    async def _redirect_get(request: Request, _new: str = _new) -> RedirectResponse:
        return RedirectResponse(url=_new, status_code=301)


@app.post("/query", include_in_schema=False)
async def _redirect_query(req: QueryRequest):
    return RedirectResponse(url="/v1/query", status_code=307)


@app.post("/reload", include_in_schema=False)
async def _redirect_reload():
    return RedirectResponse(url="/v1/reload", status_code=307)


# ── Serve UI ─────────────────────────────────────────────────────────────────

_ui_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"
)
if os.path.isdir(_ui_dir):
    app.mount("/", StaticFiles(directory=_ui_dir, html=True), name="ui")
