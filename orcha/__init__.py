"""
Orcha — Local-First Multi-Model Orchestration Engine
=====================================================

Run as many local models as you want, in parallel, and get back ONE
refined answer synthesized from all of them. No cloud API keys required.

    import asyncio
    from orcha import Orchestrator
    from orcha.experts import LocalModelRegistry

    registry = LocalModelRegistry()
    asyncio.run(registry.discover_ollama())   # finds every `ollama pull`ed model

    orc = Orchestrator(
        experts=registry.build(),
        synthesizer_expert=registry.pick_synthesizer(),
        run_all_experts=True,
    )
    result = orc.run("Explain quantum entanglement simply")
    print(result.answer)

For a zero-setup demo with no local models installed, use
`orcha.experts.mock.load_mock_experts()` instead.
"""

from .orchestrator import Orchestrator, OrchaResult
from .core.packets import OrchaPacket, PacketKind, BudgetState, ExpertResult, ExpertSlot, SubTask
from .experts.base import BaseExpert, ExpertOutput
from .experts.registry import LocalModelRegistry

__version__ = "0.3.0"

__all__ = [
    "Orchestrator", "OrchaResult",
    "OrchaPacket", "PacketKind", "BudgetState", "ExpertResult", "ExpertSlot", "SubTask",
    "BaseExpert", "ExpertOutput",
    "LocalModelRegistry",
    "__version__",
]
