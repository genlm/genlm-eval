from typing import List
import ast
import subprocess
import sys
import textwrap
import os

from genlm.control import Potential

class DS1000RuntimeNoErrorPotential(Potential):
    """
    DS-1000 expensive potential: execute the harness on a complete prefix.
    Return 0.0 if no error, -inf otherwise.
    """
    def __init__(self, vocabulary=None, code_context:str="", timeout_seconds:float=3.0):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.timeout_seconds = float(timeout_seconds)
        self.code_context = code_context

    def coerce(self, other, f=None, prune=True):
        # Overwrite coerce to adopt the LLM vocabulary without mapping tokens.
        return DS1000RuntimeNoErrorPotential(
            vocabulary=list(other.vocab),
            code_context=self.code_context,
            timeout_seconds=self.timeout_seconds,
        )

    async def prefix(self, context: List[bytes]) -> float:
        code = self._bytes_to_str(context)

        # Gate at statement boundaries (no truncation) TODO check
        #code = self._truncate_to_complete_statements(code)
        if not code.endswith("\n"):
            return 0.0
        if not self._is_complete_python(code):
            return 0.0

        return await self._score_no_error(code)

    async def complete(self, context: List[bytes]) -> float:
        return await self.prefix(context)

    def _bytes_to_str(self, toks: List[bytes]) -> str:
        return b"".join(toks).decode("utf-8", errors="ignore") if toks else ""

    def _is_complete_python(self, code: str) -> bool:
        try:
            ast.parse(code, mode="exec")
            return True
        except SyntaxError:
            return False

    def _truncate_to_complete_statements(self, code: str) -> str:
        # TODO check if we should truncate to complete statements
        s = code
        if not s.strip():
            return ""
        if self._is_complete_python(s):
            return s
        while True:
            nl = s.rfind("\n")
            if nl <= 0:
                return ""
            s = s[:nl].rstrip()
            if self._is_complete_python(s):
                return s

    async def _score_no_error(self, complete_code: str) -> float:
        if complete_code.strip() == "":
            return 0.0

        OK, BAD = "<<<OK>>>", "<<<BAD>>>"
        script = textwrap.dedent(f"""
        import sys, traceback, warnings, os
        warnings.filterwarnings("ignore")
        os.environ.setdefault("PYTHONWARNINGS", "ignore")

        code_context = {self.code_context!r}
        solution = {complete_code!r}
        OK, BAD = "<<<OK>>>", "<<<BAD>>>"

        try:
            g = {{}}
            exec(code_context, g, g)  # defines test_execution
            te = g.get("test_execution")
            if callable(te):
                try:
                    te(solution)
                    print(OK)
                except (AssertionError, KeyError, NameError):
                    # Treat harness correctness checks & missing `result` as non-fatal
                    print(OK)
                except BaseException:
                    print(BAD)
            else:
                # No harness present
                print(OK)
        except BaseException:
            print(BAD)
        """).strip()

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                check=False, capture_output=True, text=True,
                timeout=self.timeout_seconds,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return float("-inf")

        out = (proc.stdout or "") + (proc.stderr or "")
        ok = any(line.strip() == OK for line in out.splitlines())
        bad = any(line.strip().startswith(BAD) for line in out.splitlines())
        return 0.0 if ok and not bad else float("-inf")


class TrivialPotential(Potential):
    """Trivial efficient potential."""
    def __init__(self, vocabulary: List[bytes]=None):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)

    def coerce(self, other, f=None, prune=True):
        return TrivialPotential(list(other.vocab))

    async def prefix(self, context: List[bytes]) -> float:
        return 0.0

    async def complete(self, context: List[bytes]) -> float:
        return 0.0
