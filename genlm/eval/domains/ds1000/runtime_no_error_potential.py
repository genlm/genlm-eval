from typing import Dict, List, Optional
import tempfile
import subprocess
import sys
import textwrap
import os

from genlm.control import Potential
from genlm.eval.domains.ds1000.utils import (
    _sandbox_env,
    _postprocess_code,
)


class DS1000RuntimeNoErrorPotential(Potential):
    """
    DS-1000 expensive potential: execute the harness on a complete prefix.
    Return 0.0 if no error, -inf otherwise.
    """

    def __init__(
        self,
        vocabulary=None,
        code_context: str = "",
        timeout_seconds: float = 30.0,
        python_executable: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        transform_fn=None,
    ):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.timeout_seconds = float(timeout_seconds)
        self.code_context = code_context
        self.python_executable = python_executable or sys.executable
        self.extra_env = dict(extra_env or {})
        self.last_was_syntax_error = False
        self.transform_fn = transform_fn or (lambda x: x)

    def coerce(self, other, f=None, prune=True):
        if f is None:
            f = lambda x: x

        return DS1000RuntimeNoErrorPotential(
            vocabulary=list(other.vocab),
            code_context=self.code_context,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
            extra_env=self.extra_env,
            transform_fn=f,
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
        context = self.transform_fn(context)
        code = self._bytes_to_str(context)
        # Newline guardrail when using the default sampler.
        if not code.endswith("\n"):
            return 0.0
        code = _postprocess_code(code)
        out = await self._score_no_error(code)
        return out

    async def complete(self, context: List[bytes]):
        # Apply transformation before processing
        context = self.transform_fn(context)
        code = self._bytes_to_str(context)
        code = _postprocess_code(code)
        out = await self._score_no_error(code)
        return out

    async def _score_no_error(self, complete_code: str) -> float:
        """
        Run the harness script in a subprocess and check for runtime errors.
        Returns 0.0 if no error (incl. AssertionError), -inf otherwise.

        complete_code: str - the complete code to run
        Returns:
            float - 0.0 if no error, -inf otherwise.
        """
        OK, BAD, SYNTAX = "<<<OK>>>", "<<<BAD>>>", "<<<SYNTAX>>>"

        script = textwrap.dedent(
            f"""
            import sys, warnings, os, ast
            warnings.filterwarnings("ignore")
            os.environ.setdefault("PYTHONWARNINGS", "ignore")
            os.environ.setdefault("MPLBACKEND", "Agg")

            OK, BAD, SYNTAX = "<<<OK>>>", "<<<BAD>>>", "<<<SYNTAX>>>"
            code_context = {self.code_context!r}
            answer = {complete_code!r}
            try:
                g = {{}}
                exec(code_context, g, g)
                te = g.get("test_execution")
                if not callable(te):
                    print(BAD); raise SystemExit(0)
                try: # ast.parse to check if the answer is valid code
                    ast.parse(answer, filename="<answer>", mode="exec")
                except SyntaxError:
                    print(SYNTAX); raise SystemExit(0)
                try:
                    te(answer)
                    # If we get here with no exception, it ran without runtime error.
                    print(OK)
                except AssertionError:
                    print(OK)
                except SyntaxError:
                    print(SYNTAX)
                except Exception:
                    print(BAD)
            except AssertionError: # Safety check for AssertionError
                print(OK)
            except SyntaxError:
                print(SYNTAX)
            except Exception:
                print(BAD)
            """
        ).strip()

        try:
            with tempfile.TemporaryDirectory(prefix="ds1000_rt_") as td:
                path = os.path.join(td, "rt_harness.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(script + "\n")

                env = _sandbox_env(
                    td,
                    extra_env={
                        **{"MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"},
                        **self.extra_env,
                    },
                )

                proc = subprocess.run(
                    [self.python_executable, "-B", path],
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
        bad = any(line.strip() == BAD for line in out.splitlines())
        syntax = any(line.strip() == SYNTAX for line in out.splitlines())

        self.last_was_syntax_error = bool(syntax)
        bad = bad or syntax
        return 0.0 if ok and not bad else float("-inf")
