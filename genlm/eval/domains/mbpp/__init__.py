"""MBPP domain (Mostly Basic Python Problems).

Provides a dataset loader, an evaluator, and two potentials -- a runtime-no-error potential
and a test-passing potential -- over the standard Python ``assert``-based MBPP benchmark.
"""
from genlm.eval.domains.mbpp.execution import (
    MBPPRunResult,
    extract_code,
    extract_code_prefix,
    run_mbpp,
)

# The evaluator and potentials pull genlm.control -> torch/vLLM (a slow import). Load the
# heavier symbols lazily so consumers that only need execution/extraction stay cheap.
_LAZY = {
    "MBPPDataset": ".mbpp",
    "MBPPEvaluator": ".mbpp",
    "MBPPInstance": ".mbpp",
    "build_prompt": ".mbpp",
    "default_prompt_formatter": ".mbpp",
    "MBPPRuntimeNoErrorPotential": ".runtime_no_error_potential",
    "MBPPTestPassingPotential": ".test_passing_potential",
    "TestPassingFeedback": ".test_passing_potential",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)


__all__ = [
    "MBPPInstance",
    "MBPPDataset",
    "MBPPEvaluator",
    "MBPPRuntimeNoErrorPotential",
    "MBPPTestPassingPotential",
    "TestPassingFeedback",
    "MBPPRunResult",
    "build_prompt",
    "default_prompt_formatter",
    "extract_code",
    "extract_code_prefix",
    "run_mbpp",
]
