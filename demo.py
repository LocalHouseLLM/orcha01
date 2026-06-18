#!/usr/bin/env python3
"""
Orcha demo
==========
Shows Orcha running with:
  - Mock experts (zero setup, always works)
  - Real Ollama models if you have them running

Usage:
    python demo.py                          # three example queries
    python demo.py "Your question here"     # one custom query
    python demo.py --ollama                 # auto-discover local Ollama models
"""
from __future__ import annotations

import asyncio
import sys


EXAMPLES = [
    "How should I think about diversifying a small investment portfolio?",
    "Why isn't my Python async function running tasks concurrently?",
    "Why is the sky blue? Explain from first principles.",
]


def _bar(confidence: float, width: int = 20) -> str:
    filled = round(confidence * width)
    return "█" * filled + "░" * (width - filled)


def print_result(result) -> None:
    mode = "SYNTHESIZED ◈" if result.synthesized else "BEST-OF ↑"
    print(f"\n{'═'*70}")
    print(f"  {mode}  |  confidence {result.confidence:.0%}  {_bar(result.confidence)}  |  {result.iterations} iter")
    print(f"{'─'*70}")
    # word-wrap the answer at 68 chars
    words, line = result.answer.split(), ""
    for w in words:
        if len(line) + len(w) + 1 > 68:
            print("  " + line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        print("  " + line)
    print(f"{'─'*70}")
    print(f"  contributors : {', '.join(result.contributors)}")
    print(f"  primary      : {result.primary}")
    print(f"  cost         : ${result.cost:.4f}   latency: {result.latency_s:.2f}s")
    print(f"{'═'*70}\n")


async def run_mock(query: str) -> None:
    from orcha import Orchestrator
    from orcha.experts.mock import load_mock_experts

    experts = load_mock_experts()
    orc = Orchestrator(
        experts=experts,
        synthesizer_expert="mock_synthesizer",
        run_all_experts=True,
        max_iterations=2,
    )
    print(f"\n▶  {query[:80]}")
    result = await orc.run_async(query)
    print_result(result)


async def run_ollama(query: str) -> None:
    from orcha import Orchestrator
    from orcha.experts import LocalModelRegistry

    registry = LocalModelRegistry()
    await registry.discover_ollama()

    if not registry.names():
        print("  No Ollama models found — falling back to mock experts.")
        await run_mock(query)
        return

    synth = registry.pick_synthesizer()
    print(f"\n  Discovered {len(registry)} Ollama model(s): {registry.names()}")
    print(f"  Synthesizer: {synth}\n")

    orc = Orchestrator(
        experts=registry.build(),
        synthesizer_expert=synth,
        run_all_experts=True,
        max_iterations=2,
    )
    print(f"▶  {query[:80]}")
    result = await orc.run_async(query)
    print_result(result)
    print(result.explain())


async def main() -> None:
    args = sys.argv[1:]
    use_ollama = "--ollama" in args
    queries = [a for a in args if not a.startswith("--")] or EXAMPLES

    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  Orcha v0.3 — Local Multi-Model Orchestration Engine            │")
    print("└─────────────────────────────────────────────────────────────────┘")
    if use_ollama:
        print("  Mode: Ollama auto-discovery\n")
    else:
        print("  Mode: mock experts (pass --ollama to use real local models)\n")

    runner = run_ollama if use_ollama else run_mock
    for q in queries:
        await runner(q)


if __name__ == "__main__":
    asyncio.run(main())
