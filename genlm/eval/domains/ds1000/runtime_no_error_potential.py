import ast
import asyncio
import multiprocessing as mp
import os
import sys
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from genlm.control import Potential
from genlm.eval.domains.ds1000.utils import _postprocess_code


class DS1000RuntimeNoErrorPotential(Potential):
    """DS-1000 expensive potential: 0.0 if no runtime error, -inf otherwise."""

    _score_cache_maxsize = 4096

    def __init__(
        self,
        vocabulary=None,
        code_context: str = "",
        timeout_seconds: float = 30.0,
        f: Optional[Callable[[List[bytes]], List[bytes]]] = None,
        # Legacy args -- still accepted by callers, ignored by the fork runner.
        python_executable: Optional[str] = None,  # noqa: ARG002
        extra_env: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.timeout_seconds = float(timeout_seconds)
        self.code_context = code_context
        self.last_was_syntax_error = False
        self.f = f
        # Cache exact postprocessed code strings -- SMC clones often produce
        # identical prefixes.
        self._score_cache: "OrderedDict[str, tuple[float, bool]]" = OrderedDict()
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
            f=f,
        )

    def _bytes_to_str(self, toks):
        if not toks:
            return ""
        if isinstance(toks, str):
            return toks
        if isinstance(toks, bytes):
            return toks.decode("utf-8", errors="ignore")
        # Token lists may be int byte-ids or bytes; normalize before joining.
        pieces = []
        for tok in toks:
            if isinstance(tok, int):
                pieces.append(bytes([tok]))
            elif isinstance(tok, bytes):
                pieces.append(tok)
            else:
                pieces.append(str(tok).encode("utf-8", errors="ignore"))
        raw = b"".join(pieces)
        try:
            return raw.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")

    async def prefix(self, context: List[bytes]) -> float:
        if self.f is not None:
            context = self.f(context)
        code = self._bytes_to_str(context)
        # Newline guardrail: only score complete lines.
        if not code.endswith("\n"):
            return 0.0
        return await self._score_no_error(_postprocess_code(code))

    async def complete(self, context: List[bytes]):
        if self.f is not None:
            context = self.f(context)
        code = _postprocess_code(self._bytes_to_str(context))
        # Empty completion never defines ``result``; short-circuit.
        if not code:
            return float("-inf")
        return await self._score_no_error(code)

    async def _score_no_error(self, complete_code: str) -> float:
        """0.0 if the candidate runs without error (AssertionError == OK), else -inf."""
        cached = self._score_cache.get(complete_code)
        if cached is not None:
            self._score_cache.move_to_end(complete_code)
            self.cache_hits += 1
            value, syntax_error = cached
            self.last_was_syntax_error = syntax_error
            return value
        self.cache_misses += 1

        result = await _fork_score(self.code_context, complete_code, self.timeout_seconds)
        if result is None:
            # Timeout / dead worker: don't cache, the same code may succeed next time.
            self.last_was_syntax_error = False
            return float("-inf")
        ok, bad, syntax = (result == _OK), (result == _BAD), (result == _SYN)
        self.last_was_syntax_error = bool(syntax)
        value = 0.0 if (ok and not (bad or syntax)) else float("-inf")
        self._score_cache[complete_code] = (value, self.last_was_syntax_error)
        self._score_cache.move_to_end(complete_code)
        if len(self._score_cache) > self._score_cache_maxsize:
            self._score_cache.popitem(last=False)
        return value


# --- fork-per-request critic backend (parent pre-warms, child runs isolated) -

_CTX = mp.get_context("fork")
_WARM_LOCK = threading.Lock()
_WARMED = False

_OK, _BAD, _SYN = "OK", "BAD", "SYN"


def _warm() -> None:
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            __import__(mod)
        except Exception:
            pass


def _ensure_warm() -> None:
    global _WARMED
    if _WARMED:
        return
    with _WARM_LOCK:
        if not _WARMED:
            _warm()
            _WARMED = True


def _child_score(code_context: str, answer: str, conn) -> None:
    """Run inside the forked child; send the verdict back over ``conn``."""
    sys.stdout = sys.stderr = open(os.devnull, "w")  # don't leak candidate prints
    try:
        g: dict = {}
        exec(code_context, g, g)
        te = g.get("test_execution")
        if not callable(te):
            conn.send(_BAD)
            return
        try:
            ast.parse(answer, filename="<answer>", mode="exec")
        except SyntaxError:
            conn.send(_SYN)
            return
        try:
            te(answer)
            conn.send(_OK)
        except AssertionError:
            conn.send(_OK)
        except SyntaxError:
            conn.send(_SYN)
        except Exception:
            conn.send(_BAD)
    except AssertionError:
        conn.send(_OK)
    except SyntaxError:
        conn.send(_SYN)
    except Exception:
        conn.send(_BAD)
    finally:
        conn.close()


def _fork_score_sync(code_context: str, answer: str, timeout: float) -> Optional[str]:
    """Fork a child, score, return its verdict or ``None`` on timeout/failure."""
    _ensure_warm()
    parent, child = _CTX.Pipe(duplex=False)
    p = _CTX.Process(target=_child_score, args=(code_context, answer, child), daemon=True)
    p.start()
    child.close()
    try:
        if not parent.poll(timeout):
            p.kill()
            p.join(0.5)
            return None
        return parent.recv()
    except (EOFError, OSError):
        return None
    finally:
        parent.close()
        if p.is_alive():
            p.kill()
        p.join(0.5)


async def _fork_score(code_context: str, answer: str, timeout: float) -> Optional[str]:
    """Async wrapper for :func:`_fork_score_sync` (dispatched off the event loop)."""
    return await asyncio.get_running_loop().run_in_executor(
        None, _fork_score_sync, code_context, answer, timeout,
    )
