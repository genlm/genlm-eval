"""LiveCodeBench domain for genlm-eval.

The correctness signal runs the generated program against the problem's test cases
(stdin/stdout or call-based) via the vendored official LCB harness. Mirrors the
ds1000 domain (Instance / Dataset / Evaluator / Potential / prompt formatter).
"""
from .livecodebench import (
    LiveCodeBenchDataset,
    LiveCodeBenchEvaluator,
    LiveCodeBenchInstance,
    default_prompt_formatter,
)
from .prompts import extract_code, format_lcb_prompt
from .harness import check_correctness, passed_all
from ._fetch import build_row, derive_testtype
from .correctness_potential import (
    LiveCodeBenchCorrectnessPotential,
    LCBCorrectnessCritic,
    LCBTemplate,
    livecodebench_template,
)

__all__ = [
    "LiveCodeBenchInstance",
    "LiveCodeBenchDataset",
    "LiveCodeBenchEvaluator",
    "LiveCodeBenchCorrectnessPotential",
    "LCBCorrectnessCritic",
    "LCBTemplate",
    "livecodebench_template",
    "default_prompt_formatter",
    "format_lcb_prompt",
    "extract_code",
    "check_correctness",
    "passed_all",
    "build_row",
    "derive_testtype",
]
