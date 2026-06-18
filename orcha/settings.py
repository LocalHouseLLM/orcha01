"""
orcha.settings
==============
All runtime configuration in one place.
Every field can be overridden via environment variable.

    ORCHA_MAX_ITERATIONS=5      python demo.py
    OLLAMA_BASE_URL=http://192.168.1.5:11434  uvicorn orcha.api.server:app
"""
from __future__ import annotations
import os
from typing import List, Optional


def _str(k: str, d: str)   -> str:   return os.getenv(k, d)
def _int(k: str, d: int)   -> int:   return int(os.getenv(k, str(d)))
def _float(k: str, d: float) -> float: return float(os.getenv(k, str(d)))
def _bool(k: str, d: bool) -> bool:
    return os.getenv(k, str(d)).lower() in ("1", "true", "yes", "on")


class OrchaSettings:
    """
    Runtime settings. Access via the module-level `settings` singleton.

    Environment variable mapping (all optional):
    --------------------------------------------
    ORCHA_MAX_COST             float   default 1.0
    ORCHA_MAX_LATENCY_S        float   default 120.0
    ORCHA_MAX_ITERATIONS       int     default 3
    ORCHA_QUALITY_THRESHOLD    float   default 0.70
    ORCHA_RUN_ALL_EXPERTS      bool    default false
    ORCHA_USE_EMBEDDINGS       bool    default false
    ORCHA_API_HOST             str     default 0.0.0.0
    ORCHA_API_PORT             int     default 8420
    OLLAMA_BASE_URL            str     default http://localhost:11434
    OLLAMA_AUTO_DISCOVER       bool    default true
    OLLAMA_SKIP_PATTERNS       str     comma-sep substrings to skip, default ""
    LOCAL_SERVER_BASE_URL      str     default http://localhost:8080/v1
    LOCAL_SERVER_MODEL         str     default ""
    """

    # ── Budget defaults ────────────────────────────────────────────────
    max_cost:           float = _float("ORCHA_MAX_COST",         1.0)
    max_latency_s:      float = _float("ORCHA_MAX_LATENCY_S",  120.0)
    max_iterations:     int   = _int(  "ORCHA_MAX_ITERATIONS",    3)
    quality_threshold:  float = _float("ORCHA_QUALITY_THRESHOLD", 0.70)

    # ── Routing ────────────────────────────────────────────────────────
    run_all_experts: bool = _bool("ORCHA_RUN_ALL_EXPERTS", False)
    use_embeddings:  bool = _bool("ORCHA_USE_EMBEDDINGS",  False)

    # ── Ollama ────────────────────────────────────────────────────────
    ollama_base_url:     str  = _str("OLLAMA_BASE_URL",    "http://localhost:11434")
    ollama_auto_discover: bool = _bool("OLLAMA_AUTO_DISCOVER", True)
    ollama_skip_patterns: List[str] = [
        s.strip()
        for s in _str("OLLAMA_SKIP_PATTERNS", "").split(",")
        if s.strip()
    ]

    # ── OpenAI-compatible local server ────────────────────────────────
    local_server_base_url: str = _str("LOCAL_SERVER_BASE_URL", "http://localhost:8080/v1")
    local_server_model:    str = _str("LOCAL_SERVER_MODEL",    "")

    # ── Web UI / API ──────────────────────────────────────────────────
    api_host:  str = _str("ORCHA_API_HOST", "0.0.0.0")
    api_port:  int = _int("ORCHA_API_PORT",  8420)

    def __repr__(self) -> str:
        return (
            f"OrchaSettings(max_iterations={self.max_iterations}, "
            f"max_cost={self.max_cost}, "
            f"ollama={self.ollama_base_url}, "
            f"run_all={self.run_all_experts})"
        )


settings = OrchaSettings()
