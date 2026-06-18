# Orcha

**Local-first multi-model orchestration engine.**

Orcha runs as many local LLMs as you want **in parallel**, then uses the
largest one to **synthesize** every answer into a single, refined response —
no cloud API needed, no vendor lock-in.

```
your query
    │
    ▼
Decompose ──► detect domain (finance, code, reasoning, science…)
    │
    ▼
Select ──────► score every local model against the domain
    │
    ▼
Execute ────► llama3:8b ──┐
             mistral:7b ──┤  all run concurrently (asyncio)
             codellama ───┤
             qwen2.5:32b ─┘
    │
    ▼
Synthesize ──► qwen2.5:70b reads every candidate and writes ONE refined answer
    │
    ▼
Evaluate ────► confidence check → retry if below threshold
    │
    ▼
  answer
```

Everything is wired through a single typed `OrchaPacket` message that
every stage reads, writes back to, and stamps with a trace entry — so you
always know exactly what happened and why.

---

## Quickstart

```bash
git clone https://github.com/LocalHouseLLM/orcha01
cd orcha01
pip install -e ".[all]"          # install everything
```

**With mock experts (zero setup, works immediately):**

```python
from orcha import Orchestrator
from orcha.experts.mock import load_mock_experts

orc = Orchestrator(
    experts=load_mock_experts(),
    synthesizer_expert="mock_synthesizer",
    run_all_experts=True,
)
result = orc.run("How should I diversify a small portfolio?")
print(result.answer)
print(result.contributors)   # which models answered
print(result.synthesized)    # True = answer was synthesized by a local model
```

**With real Ollama models:**

```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull qwen2.5:32b     # will become the synthesizer
```

```python
import asyncio
from orcha import Orchestrator, LocalModelRegistry

async def main():
    registry = LocalModelRegistry()
    await registry.discover_ollama()          # finds every pulled model automatically

    orc = Orchestrator(
        experts=registry.build(),
        synthesizer_expert=registry.pick_synthesizer(),   # largest model wins
        run_all_experts=True,
    )
    result = await orc.run_async("Explain quantum entanglement simply.")
    print(result.answer)

asyncio.run(main())
```

**Web UI:**

```bash
uvicorn orcha.api.server:app --reload --port 8420
# open http://localhost:8420
```

> **API versioning:** The stable HTTP API lives under `/v1` (e.g. `POST /v1/query`,
> `GET /v1/health`). Un-versioned paths (`/query`, `/health`) redirect to `/v1` for
> backwards compatibility and are deprecated.

---

## Installation

```bash
pip install -e ".[all]"      # everything (recommended)
pip install -e "."           # core only (no HTTP, no YAML)
pip install -e ".[local]"    # + httpx for Ollama / local servers
pip install -e ".[api]"      # + FastAPI web UI
pip install -e ".[config]"   # + PyYAML for config files
pip install -e ".[dev]"      # + pytest for development
```

Requires Python 3.10+. No cloud API key is ever required.

---

## Adding models

### Ollama (recommended)

```python
from orcha.experts import LocalModelRegistry

registry = LocalModelRegistry()

# Auto-discover everything in `ollama list`
import asyncio
asyncio.run(registry.discover_ollama())

# Or add specific models with metadata
registry.add_ollama("llama3.1:8b",  domain="general")
registry.add_ollama("codellama:13b", domain="code")
registry.add_ollama("llama3.1:70b", synthesizer=True)   # explicit synthesizer

pool  = registry.build()
synth = registry.pick_synthesizer()

orc = Orchestrator(experts=pool, synthesizer_expert=synth, run_all_experts=True)
```

### Any OpenAI-compatible local server

Works with **llama.cpp**, **LM Studio**, **vLLM**, **text-generation-webui**,
**LocalAI**, **Jan**, and anything else that serves
`POST /v1/chat/completions`.

```python
from orcha.experts import LocalChatExpert

expert = LocalChatExpert(
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:8080/v1",   # llama.cpp --port 8080
    domain="reasoning",
    description="32B reasoning model via llama.cpp",
)
orc.register_expert("qwen32b", expert)
```

### Config file (JSON or YAML)

```json
{
  "models": [
    {"backend": "ollama", "model": "llama3:8b",   "domain": "general"},
    {"backend": "ollama", "model": "codellama:13b", "domain": "code"},
    {"backend": "ollama", "model": "llama3:70b",   "domain": "reasoning", "synthesizer": true},
    {
      "backend": "openai_compatible",
      "model": "qwen2.5-32b-instruct",
      "base_url": "http://localhost:8080/v1",
      "domain": "reasoning"
    }
  ]
}
```

```python
registry = LocalModelRegistry.from_config("models.json")
orc = Orchestrator(
    experts=registry.build(),
    synthesizer_expert=registry.pick_synthesizer(),
    run_all_experts=True,
)
```

### Write your own expert

```python
from orcha.experts.base import BaseExpert, ExpertOutput

class MyExpert(BaseExpert):
    name        = "my_model"
    domain      = "finance"           # used for routing
    description = "Custom fine-tuned finance model"

    async def execute(self, query: str) -> ExpertOutput:
        answer = await my_inference_call(query)
        return ExpertOutput(answer=answer, confidence=0.85, tokens_used=200)

orc.register_expert("my_model", MyExpert())
```

---

## How the synthesizer works

When `synthesizer_expert` is configured and at least two models produced
usable answers, the aggregator builds this prompt and hands it to the
synthesizer:

```
You are a synthesis expert. Several specialist models independently answered
the question below. Combine their insights into ONE refined, accurate,
well-organized final answer. Resolve disagreements by favoring the
better-reasoned position. Do not simply concatenate the answers.

Question:
{query}

Candidate answers:
--- Candidate 1 (llama3:8b) ---
{answer from llama3}

--- Candidate 2 (mistral:7b) ---
{answer from mistral}

Final answer:
```

The synthesizer (typically the largest model in your pool) then writes one
clean, reconciled response. If synthesis fails for any reason, the system
falls back to confidence-weighted aggregation automatically.

---

## Architecture

| File | Purpose |
|---|---|
| `orcha/core/packets.py` | `OrchaPacket` — the typed message flowing through every stage |
| `orcha/orchestrator.py` | Main loop: decompose → select → execute → aggregate → evaluate → retry |
| `orcha/orchestration/decomposer.py` | Domain detection, subtask decomposition |
| `orcha/orchestration/planner.py` | Budget-aware effort scaling per iteration |
| `orcha/orchestration/selector.py` | Domain-match scoring, `force_all_experts` mode |
| `orcha/orchestration/executor.py` | `asyncio.gather` parallel execution, fault isolation |
| `orcha/orchestration/aggregator.py` | Synthesis prompt + confidence-weighted fallback |
| `orcha/orchestration/evaluator.py` | Quality gate; triggers retry |
| `orcha/orchestration/retry.py` | Retry decision vs budget |
| `orcha/experts/ollama.py` | Ollama backend |
| `orcha/experts/local_chat.py` | OpenAI-compatible local server backend |
| `orcha/experts/registry.py` | `LocalModelRegistry` — discover, configure, pick synthesizer |
| `orcha/experts/mock.py` | Zero-dependency mock experts + `MockSynthesizer` |
| `orcha/api/server.py` | FastAPI server, Ollama auto-discovery on startup (lifespan), versioned `/v1` routes |
| `orcha/observability.py` | Structured logging with trace IDs |
| `orcha/ui/index.html` | Web dashboard |

---

## Tests

```bash
pip install -e ".[dev]"
pytest -v                  # 86 tests
```

---

## License

MIT
