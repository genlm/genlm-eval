"""LiveCodeBench correctness Potential + identity prompt template (from latent critic.py).

Terminal 0/1 constraint for SMC: ``complete`` -> 0.0 if the decoded code passes all
tests else -inf; ``prefix`` -> 0.0. Needs genlm-control; only for constrained generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from genlm.control.constant import EndOfSequence
from genlm.control.potential import Potential

from genlm.eval.domains.livecodebench.harness import passed_all
from genlm.eval.domains.livecodebench.prompts import extract_code


@dataclass
class LCBTemplate:
    """Identity template: the formatted prompt string is built upstream (via
    ``default_prompt_formatter`` / ``format_lcb_prompt``), so this passes it through."""
    format_prompt: Callable[[str], str]
    format_prompt_ids: Optional[Callable[[str], List[int]]] = None


def livecodebench_template(tokenizer=None) -> LCBTemplate:
    fmt = lambda prompt: prompt
    if tokenizer is None:
        return LCBTemplate(format_prompt=fmt)
    fmt_ids = lambda prompt: list(tokenizer.encode(prompt))
    return LCBTemplate(format_prompt=fmt, format_prompt_ids=fmt_ids)


class LiveCodeBenchCorrectnessPotential(Potential):
    """0/1 terminal correctness constraint for LiveCodeBench.

    ``complete`` returns ``0.0`` iff the decoded particle's extracted code passes
    every test in ``eval_sample``; ``-inf`` otherwise. ``prefix`` returns ``0.0``
    (no in-rollout pruning)."""

    def __init__(self, vocab, eval_sample: Dict[str, str], timeout_seconds: float = 6.0):
        super().__init__(vocabulary=vocab)
        self.eval_sample = eval_sample
        self.timeout_seconds = float(timeout_seconds)

    async def _passes(self, context) -> float:
        bytes_only = [t for t in context if not isinstance(t, EndOfSequence)]
        try:
            text = b"".join(bytes_only).decode("utf-8")
        except UnicodeDecodeError:
            return float("-inf")
        code = extract_code(text)
        ok = passed_all(self.eval_sample, code, timeout=self.timeout_seconds)
        return 0.0 if ok else float("-inf")

    async def complete(self, context) -> float:
        return await self._passes(context)

    async def prefix(self, context) -> float:
        return 0.0


# Back-compat alias matching the latent PR name.
LCBCorrectnessCritic = LiveCodeBenchCorrectnessPotential
