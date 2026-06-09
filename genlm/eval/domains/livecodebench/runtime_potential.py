"""Runtime no-error potential for LiveCodeBench (mirrors DS1000RuntimeNoErrorPotential).

Runs a generation on a test input; 0.0 if it executes without raising, -inf otherwise.
Uses the input only (never the gold output), so wrong answers still score 0.0 — only
broken code is pruned. Correctness is the Evaluator's job.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

from genlm.control import Potential

from genlm.eval.domains.livecodebench.util.prompts import extract_code


class LiveCodeBenchRuntimeNoErrorPotential(Potential):
    """Run a complete generation's code on a test input; 0.0 if no error, -inf otherwise."""

    def __init__(
        self,
        vocabulary=None,
        eval_sample: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 10.0,
        python_executable: Optional[str] = None,
        f: Optional[Callable[[List[bytes]], List[bytes]]] = None,
    ):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.timeout_seconds = float(timeout_seconds)
        self.python_executable = python_executable or sys.executable
        self.f = f
        self._stdin = self._first_input(eval_sample or {})  # keeps stdin programs from crashing
        self._eval_sample = dict(eval_sample or {})
        # cache exact code strings (SMC clones particles into repeated prefixes)
        self._score_cache: "OrderedDict[str, float]" = OrderedDict()
        self._score_cache_maxsize = 4096
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _first_input(eval_sample: Dict[str, str]) -> str:
        try:
            inputs = json.loads(eval_sample["input_output"])["inputs"]
            return inputs[0] if inputs else ""
        except (KeyError, ValueError, IndexError, TypeError):
            return ""

    def coerce(self, other, f: Optional[Callable[[List[bytes]], List[bytes]]] = None, prune: bool = True):
        return LiveCodeBenchRuntimeNoErrorPotential(
            vocabulary=list(other.vocab),
            eval_sample=self._eval_sample,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
            f=f,
        )

    def _bytes_to_str(self, toks) -> str:
        if not toks:
            return ""
        if isinstance(toks, str):
            return toks
        if isinstance(toks, bytes):
            return toks.decode("utf-8", errors="ignore")
        # genlm-control/vLLM contexts are byte tokens or integer byte ids; skip any
        # sentinels (e.g. EndOfSequence) so they don't corrupt the decoded program.
        pieces = []
        for tok in toks:
            if isinstance(tok, int):
                pieces.append(bytes([tok]))
            elif isinstance(tok, bytes):
                pieces.append(tok)
        return b"".join(pieces).decode("utf-8", errors="ignore")

    async def prefix(self, context: List[bytes]) -> float:
        return 0.0  # no in-rollout pruning; the check is only meaningful on a full program

    async def complete(self, context: List[bytes]) -> float:
        if self.f is not None:
            context = self.f(context)
        code = extract_code(self._bytes_to_str(context))
        if not code:
            return float("-inf")
        return await self._score_no_error(code)

    async def _score_no_error(self, code: str) -> float:
        cached = self._score_cache.get(code)
        if cached is not None:
            self._score_cache.move_to_end(code)
            self.cache_hits += 1
            return cached
        self.cache_misses += 1

        OK, BAD = "<<<OK>>>", "<<<BAD>>>"
        script = textwrap.dedent(
            f"""
            import sys, ast, warnings, os
            warnings.filterwarnings("ignore")
            os.environ.setdefault("MPLBACKEND", "Agg")
            OK, BAD = "<<<OK>>>", "<<<BAD>>>"
            answer = {code!r}
            try:
                ast.parse(answer, filename="<answer>", mode="exec")
            except SyntaxError:
                print(BAD); raise SystemExit(0)
            try:
                exec(compile(answer, "<answer>", "exec"), {{"__name__": "__main__"}})
                print(OK)
            except (EOFError, SystemExit):   # end-of-input / sys.exit() = clean run
                print(OK)
            except BaseException:
                print(BAD)
            """
        ).strip()

        value = float("-inf")
        try:
            with tempfile.TemporaryDirectory(prefix="lcb_rt_") as td:
                path = os.path.join(td, "rt_harness.py")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(script + "\n")
                env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
                       "PYTHONWARNINGS": "ignore"}
                proc = await asyncio.create_subprocess_exec(
                    self.python_executable, "-B", path,
                    stdin=subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    env=env, cwd=td,
                )
                try:
                    stdout_b, _ = await asyncio.wait_for(
                        proc.communicate(input=self._stdin.encode("utf-8", "ignore")),
                        timeout=self.timeout_seconds)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    return float("-inf")
                lines = {ln.strip() for ln in stdout_b.decode("utf-8", "replace").splitlines()}
                value = 0.0 if (OK in lines and BAD not in lines) else float("-inf")
        except (OSError, subprocess.SubprocessError):
            return float("-inf")

        self._score_cache[code] = value
        self._score_cache.move_to_end(code)
        if len(self._score_cache) > self._score_cache_maxsize:
            self._score_cache.popitem(last=False)
        return value
