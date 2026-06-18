"""
orcha.experts.local_chat
=========================
Expert for ANY locally-hosted OpenAI-compatible chat endpoint.

Works out of the box with
--------------------------
llama.cpp server   ./llama-server -m model.gguf --port 8080
LM Studio          starts on localhost:1234 by default
vLLM               python -m vllm.entrypoints.openai.api_server ...
text-gen-webui     with the openai extension enabled
LocalAI            drop-in OpenAI replacement
Jan                local AI assistant with server mode
Kobold.cpp         with OpenAI-compat mode enabled

No cloud account, no API key, no internet connection required.
The api_key field defaults to "not-needed" — set it to anything
if the local server requires a token (some do for access control).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .base import BaseExpert, ExpertOutput


def _slugify(s: str) -> str:
    return (
        s.replace(":", "_").replace(".", "_")
         .replace("/", "_").replace("-", "_").replace(" ", "_")
    )


class LocalChatExpert(BaseExpert):
    """
    Generic expert for any OpenAI-compatible local inference server.
    Talks to POST /v1/chat/completions.
    """

    version            = "1.1"
    cost_per_1k_tokens = 0.0   # local inference — no per-token cost

    def __init__(
        self,
        model:          str,
        base_url:       str           = "http://localhost:8080/v1",
        name:           Optional[str] = None,
        domain:         str           = "general",
        description:    Optional[str] = None,
        system_prompt:  Optional[str] = None,
        temperature:    float         = 0.7,
        top_p:          float         = 0.9,
        max_tokens:     int           = 1024,
        api_key:        str           = "not-needed",
        timeout_s:      float         = 240.0,
        extra_body:     Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        model           Model identifier as the server knows it.
        base_url        Root of the OpenAI-compatible API (must end before /chat).
        name            Expert name in Orcha's registry. Auto-derived if None.
        domain          Routing domain (general, code, reasoning, …).
        description     Human-readable description used by the selector.
        system_prompt   Optional system message prepended to every call.
        temperature     Sampling temperature (0 = greedy, 1 = creative).
        top_p           Nucleus sampling threshold.
        max_tokens      Maximum tokens in the generated response.
        api_key         Bearer token. Most local servers ignore this.
        timeout_s       Per-call HTTP timeout.
        extra_body      Any additional JSON fields to include in the request
                        body (e.g. llama.cpp-specific options).
        """
        self.model         = model
        self.base_url      = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature   = temperature
        self.top_p         = top_p
        self.max_tokens    = max_tokens
        self.api_key       = api_key
        self.timeout_s     = timeout_s
        self.extra_body    = extra_body or {}

        self.name        = name or f"local_{_slugify(model)}"
        self.domain      = domain
        self.description = description or f"Local model '{model}' at {base_url}"

    async def execute(self, query: str) -> ExpertOutput:
        import httpx  # soft dep

        t0 = time.perf_counter()
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})

        body: Dict[str, Any] = {
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "top_p":       self.top_p,
            "max_tokens":  self.max_tokens,
            "stream":      False,
            **self.extra_body,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        choice         = data["choices"][0]
        answer         = (choice.get("message", {}).get("content") or "").strip()
        finish_reason  = choice.get("finish_reason", "stop")
        usage          = data.get("usage") or {}
        tokens         = usage.get("total_tokens", 0)
        latency_s      = time.perf_counter() - t0

        # Confidence proxy: natural stop > length-truncated > other
        if finish_reason == "stop":
            confidence = 0.80
        elif finish_reason == "length":
            confidence = 0.62  # answer may be truncated
        else:
            confidence = 0.65

        return ExpertOutput(
            answer=answer,
            confidence=confidence,
            tokens_used=tokens,
            latency_s=latency_s,
            finish_reason=finish_reason,
            model_version=data.get("model", self.model),
            metadata={
                "prompt_tokens":     usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        )

    async def healthcheck(self) -> bool:
        """
        Checks the /models endpoint. Returns True if it responds with 200,
        even if the specific model is not listed (some servers don't enumerate).
        """
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return resp.status_code == 200
        except Exception:
            return False

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "model":       self.model,
            "base_url":    self.base_url,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        })
        return base
