from typing import List
import tempfile
import subprocess
import sys
import textwrap
import os

from genlm.control import Potential
from genlm.eval.domains.ds1000.utils import _sandbox_env, _postprocess_code

class DS1000RuntimeNoErrorPotential(Potential):
    """
    DS-1000 expensive potential: execute the harness on a complete prefix.
    Return 0.0 if no error, -inf otherwise.
    """
    def __init__(self, vocabulary=None, code_context:str="", timeout_seconds:float=30.0):
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
    
    def _bytes_to_str(self, toks):
        if not toks:
            return ""
        try:
            bytes_str = b"".join(toks).decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            bytes_str = b"".join(toks).decode("latin-1", errors="ignore")
        return bytes_str

    async def prefix(self, context: List[bytes]) -> float:
        code = self._bytes_to_str(context)
        if not code.endswith("\n"):
            return 0.0
        code = _postprocess_code(code)
        out = await self._score_no_error(code)
        return out
    
    async def complete(self, context):
        return await self.prefix(context)    

    async def _score_no_error(self, complete_code: str) -> float:
        if complete_code.strip() == "":
            return 0.0
        OK, BAD = "<<<OK>>>", "<<<BAD>>>"
        script = textwrap.dedent(f"""
        import sys, traceback, warnings, os
        warnings.filterwarnings("ignore")
        os.environ.setdefault("PYTHONWARNINGS", "ignore")
        os.environ.setdefault("MPLBACKEND", "Agg")

        code_context = {self.code_context!r}
        solution = {complete_code!r}
        OK, BAD = "<<<OK>>>", "<<<BAD>>>"

        try:
            g = {{}}
            exec(code_context, g, g)  # defines test_execution(solution)
            te = g.get("test_execution")
            if not callable(te):
                print(BAD); raise SystemExit(0)

            try:
                te(solution)
                # If we get here with no exception, it ran without runtime error.
                print(OK)
            except AssertionError:
                # Treat harness correctness checks & missing `result` as non-fatal
                print(OK)
            except BaseException:
                print(BAD)
        except BaseException:
            print(BAD)
        """).strip()

        try:
            with tempfile.TemporaryDirectory(prefix="ds1000_rt_") as td:
                path = os.path.join(td, "rt_harness.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(script + "\n")

                env = _sandbox_env(td, extra_env={"MPLBACKEND": "Agg"})
                proc = subprocess.run(
                    [sys.executable, "-B", path],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=env,
                    cwd=td,
                )
        except subprocess.TimeoutExpired:
            return float("-inf")

        out = (proc.stdout or "") + (proc.stderr or "")
        ok = any(line.strip() == OK for line in out.splitlines())
        bad = any(line.strip().startswith(BAD) for line in out.splitlines())
        return 0.0 if ok and not bad else float("-inf")
