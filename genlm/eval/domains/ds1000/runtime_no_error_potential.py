from typing import Callable, Dict, List, Optional
import asyncio
from collections import OrderedDict
import tempfile
import subprocess
import sys
import textwrap
import os

from genlm.control import Potential
from genlm.eval.domains.ds1000.utils import _postprocess_code, _sandbox_env


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
        f: Optional[Callable[[List[bytes]], List[bytes]]] = None,
    ):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.timeout_seconds = float(timeout_seconds)
        self.code_context = code_context
        self.python_executable = python_executable or sys.executable
        self.extra_env = dict(extra_env or {})
        self.last_was_syntax_error = False
        self.f = f
        # SMC frequently clones particles into identical or repeated prefixes.
        # Cache exact postprocessed code strings to avoid launching duplicate
        # DS1000 subprocess checks. The cache is per potential instance, whose
        # code_context / timeout / environment are fixed.
        self._score_cache = OrderedDict()
        self._score_cache_maxsize = 4096
        self.cache_hits = 0
        self.cache_misses = 0

    def coerce(
        self,
        other,
        f: Optional[Callable[[List[bytes]], List[bytes]]] = None,
        prune: bool = True,
    ):
        return DS1000RuntimeNoErrorPotential(
            vocabulary=list(other.vocab),
            code_context=self.code_context,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
            extra_env=self.extra_env,
            f=f,
        )

    def _bytes_to_str(self, toks):
        if not toks:
            return ""
        if isinstance(toks, str):
            return toks
        if isinstance(toks, bytes):
            return toks.decode("utf-8", errors="ignore")
        # genlm-control/vLLM contexts may be either byte tokens or integer byte ids.
        # Normalize both forms before decoding so the potential is robust across
        # backend/token-map representations.
        byte_pieces = []
        for tok in toks:
            if isinstance(tok, int):
                byte_pieces.append(bytes([tok]))
            elif isinstance(tok, bytes):
                byte_pieces.append(tok)
            else:
                byte_pieces.append(str(tok).encode("utf-8", errors="ignore"))
        raw = b"".join(byte_pieces)
        try:
            bytes_str = raw.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            bytes_str = raw.decode("latin-1", errors="ignore")
        return bytes_str

    async def prefix(self, context: List[bytes]) -> float:
        if self.f is not None:
            context = self.f(context)
        code = self._bytes_to_str(context)
        # Newline guardrail when using the default sampler.
        if not code.endswith("\n"):
            return 0.0
        code = _postprocess_code(code)
        out = await self._score_no_error(code)
        return out

    async def complete(self, context: List[bytes]):
        # Apply transformation before processing
        if self.f is not None:
            context = self.f(context)
        code = self._bytes_to_str(context)
        code = _postprocess_code(code)
        # Empty completion don't define `result`, so the harness will
        # always raise KeyError. Skip the subprocess and return -inf directly.
        # The LM often emits EOS / "</code>" as the first token.
        if not code:
            return float("-inf")
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
        cached = self._score_cache.get(complete_code)
        if cached is not None:
            self._score_cache.move_to_end(complete_code)
            self.cache_hits += 1
            value, syntax_error = cached
            self.last_was_syntax_error = syntax_error
            return value
        self.cache_misses += 1

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

                proc = await asyncio.create_subprocess_exec(
                    self.python_executable,
                    "-B",
                    path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=td,
                )
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(
                        proc.communicate(), timeout=self.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    return float("-inf")
        except subprocess.TimeoutExpired:
            return float("-inf")

        out = stdout_b.decode("utf-8", errors="replace") + stderr_b.decode(
            "utf-8", errors="replace"
        )
        ok = any(line.strip() == OK for line in out.splitlines())
        bad = any(line.strip() == BAD for line in out.splitlines())
        syntax = any(line.strip() == SYNTAX for line in out.splitlines())

        self.last_was_syntax_error = bool(syntax)
        bad = bad or syntax
        value = 0.0 if ok and not bad else float("-inf")
        self._score_cache[complete_code] = (value, self.last_was_syntax_error)
        self._score_cache.move_to_end(complete_code)
        if len(self._score_cache) > self._score_cache_maxsize:
            self._score_cache.popitem(last=False)
        return value
