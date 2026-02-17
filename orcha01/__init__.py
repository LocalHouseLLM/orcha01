"""
Orcha 0.1 — Modular Multi-Expert Orchestration Engine.

Public API surface.
Everything exported here is considered stable.
"""

from importlib.metadata import version, PackageNotFoundError

# ============================================================
# Version
# ============================================================

try:
    __version__ = version("orcha01")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


# ============================================================
# Core Entry
# ============================================================

from .orchestrator import Orchestrator


# ============================================================
# Orchestration Components
# ============================================================

from .orchestration.decomposer import Decomposer, get_decomposer
from .orchestration.selector import Selector, get_selector
from .orchestration.executor import (
    Executor,
    ExecutionRequest,
    ExecutionResult,
)
from .orchestration.aggregator import Aggregator, get_aggregator
from .orchestration.evaluator import Evaluator, get_evaluator
from .orchestration.retry import RetryController, get_retry_controller


# ============================================================
# Experts
# ============================================================

from .experts.base import BaseExpert


# ============================================================
# What users can import via:
# from orcha01 import *
# ============================================================

__all__ = [
    "__version__",
    "Orchestrator",

    "Decomposer",
    "get_decomposer",

    "Selector",
    "get_selector",

    "Executor",
    "ExecutionRequest",
    "ExecutionResult",

    "Aggregator",
    "get_aggregator",

    "Evaluator",
    "get_evaluator",

    "RetryController",
    "get_retry_controller",

    "BaseExpert",
]
