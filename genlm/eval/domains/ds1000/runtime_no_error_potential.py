from typing import Callable, Dict, List, Optional
import asyncio
import ast
import codeop
from collections import OrderedDict
import re
import tempfile
import subprocess
import sys
import textwrap
import os
import uuid

from genlm.control import Potential
from genlm.eval.domains.ds1000.forkserver import ForkserverUnavailable, shared_executor
from genlm.eval.domains.ds1000.utils import _postprocess_official, _sandbox_env

# Once one of these appears in the raw generation, _postprocess_official truncates
# there and no continuation can change the solution anymore.
_TERMINAL_MARKERS = ("</code>", "\nEND SOLUTION")

# Bound concurrent subprocess spawns per event loop: asyncio's default child
# watcher starts one OS thread per subprocess, so an unbounded fallback storm
# (e.g. fork-server down + high caller concurrency) exhausts the thread limit.
_subprocess_sems = {}


def _subprocess_sem():
    loop = asyncio.get_running_loop()
    sem = _subprocess_sems.get(id(loop))
    if sem is None:
        n = int(os.environ.get("DS1000_SUBPROCESS_CONCURRENCY", "8"))
        sem = asyncio.Semaphore(n)
        _subprocess_sems[id(loop)] = sem
    return sem

# Single source of truth for prefix judging, embedded verbatim into every
# harness script (stateless and session) so the paths cannot drift. Defines
# `_judge_prefix`, returning one of the OK/BAD/SYNTAX markers. chr(10)/chr(92)
# stand in for newline/backslash to keep this free of escape sequences when
# emitted into a generated script.
_PREFIX_JUDGE_SRC = '''
def _judge_prefix(answer, head, skip, test_inputs, OK, BAD, SYNTAX):
    import ast
    _nl = chr(10)
    try:
        solution = answer
        # Matplotlib harnesses drop plt.show()/savefig/... before executing.
        if callable(skip):
            solution = _nl.join(filter(skip, solution.split(_nl)))
        # Trailing line-continuation: a later boundary completes it -> defer.
        if solution.rstrip().endswith(chr(92)):
            return OK
        try:
            tree = ast.parse(head + solution, filename="<prefix>", mode="exec")
        except SyntaxError:
            return SYNTAX
        # A trailing compound statement may still be extended: do not run it.
        if tree.body and hasattr(tree.body[-1], "body"):
            tree.body = tree.body[:-1]
        # Head statements (head alone may not parse, e.g. a dangling def) are
        # harness-side; the rest is the solution.
        head_lines = head.count(_nl)
        n_head = sum(1 for st in tree.body if st.lineno <= head_lines)
        head_prog = compile(
            ast.Module(body=tree.body[:n_head], type_ignores=[]), "<prefix>", "exec")
        sol_prog = compile(
            ast.Module(body=tree.body[n_head:], type_ignores=[]), "<prefix>", "exec")
        for ti in test_inputs:
            test_env = {"test_input": ti}
            try:
                exec(head_prog, test_env)
            except Exception:
                # Test-environment setup failure is not the solution's fault.
                return OK
            exec(sol_prog, test_env)
        return OK
    except AssertionError:
        return OK
    except SyntaxError:
        return SYNTAX
    except Exception:
        return BAD
'''


def _prefix_syntax_status(code: str) -> str:
    """
    Classify a solution prefix: "complete" (parses), "incomplete" (a
    continuation can still fix it), or "broken" (no continuation can).
    """
    # codeop misclassifies a trailing line-continuation backslash as broken.
    if code.endswith("\\\n"):
        return "incomplete"
    try:
        ast.parse(code)
        return "complete"
    except SyntaxError:
        pass
    try:
        if codeop.compile_command(code, "<prefix>", "exec") is None:
            return "incomplete"
        # codeop compiled what ast.parse rejected; be conservative and defer.
        return "incomplete"
    except (SyntaxError, ValueError, OverflowError):
        return "broken"


def _test_case_count(code_context: str) -> int:
    """Number of test cases iterated by test_execution in the code context."""
    m = re.search(
        r"def test_execution.*?for\s+i\s+in\s+range\((\d+)\)", code_context, re.S
    )
    return int(m.group(1)) if m else 1


def _exec_context_head(code_context: str):
    """
    Return (head, is_block_insert): exec_context before the `[insert]` line
    (None if absent), and whether `[insert]` is on its own line (vs inline,
    e.g. `[insert].numpy()`).
    """
    m = re.search(r"exec_context\s*=\s*r?(\"\"\"|''')(.*?)\1", code_context, re.S)
    if not m:
        return None, False
    exec_context = m.group(2)
    lines = exec_context.split("\n")
    for i, line in enumerate(lines):
        if "[insert]" in line:
            head = "\n".join(lines[:i])
            if head:
                head += "\n"
            return head, line == "[insert]"
    return None, False


class DS1000RuntimeNoErrorPotential(Potential):
    """
    DS-1000 potential: 0.0 if the harness raises no error, -inf otherwise.
    prefix() runs exec_context-head + partial solution (no answer checks);
    complete() runs the full test_execution harness.
    """

    def __init__(
        self,
        vocabulary=None,
        code_context: str = "",
        timeout_seconds: float = 30.0,
        python_executable: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        f: Optional[Callable[[List[bytes]], List[bytes]]] = None,
        strict_prefix: bool = False,
        use_forkserver: bool = True,
    ):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.timeout_seconds = float(timeout_seconds)
        self.code_context = code_context
        self.python_executable = python_executable or sys.executable
        self.extra_env = dict(extra_env or {})
        self.last_was_syntax_error = False
        self.f = f
        # strict_prefix=True scores prefixes with the full harness (answer
        # checks included) instead of the head-only check.
        self.strict_prefix = bool(strict_prefix)
        # Warm fork-server (default), same verdicts as the subprocess path but
        # faster, with subprocess fallback. DS1000_FORKSERVER=0 is the ops
        # kill-switch; startup env vars (e.g. PYTHONHASHSEED) apply at worker start.
        if os.environ.get("DS1000_FORKSERVER") == "0":
            use_forkserver = False
        # TensorFlow is unsafe to import in a forked child (it is the one lib
        # the worker does not pre-import); such checks fork-then-import TF and
        # mis-score. Force the subprocess path (fresh interpreter) for them.
        self._fork_unsafe = "import tensorflow" in code_context
        self.use_forkserver = bool(use_forkserver) and not self._fork_unsafe
        self._n_test_cases = _test_case_count(code_context)
        self._ec_head, self._ec_block_insert = _exec_context_head(code_context)
        self._session_key = "ds1000-" + uuid.uuid5(
            uuid.NAMESPACE_OID, code_context
        ).hex
        # Prefixes whose check timed out: extensions re-run the same leading
        # statements, so defer them without paying the timeout again.
        self._hung_prefixes = []
        self._hung_prefixes_maxsize = 16
        # SMC clones particles into repeated prefixes; cache verdicts per
        # exact postprocessed code (instance config is fixed).
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
            strict_prefix=self.strict_prefix,
            use_forkserver=self.use_forkserver,
        )

    def _bytes_to_str(self, toks):
        if not toks:
            return ""
        if isinstance(toks, str):
            return toks
        if isinstance(toks, bytes):
            return toks.decode("utf-8", errors="ignore")
        # Contexts may be byte tokens or integer byte ids; normalize both.
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
        raw = self._bytes_to_str(context)
        # Newline guardrail when using the default sampler.
        if not raw.endswith("\n"):
            return 0.0
        code = _postprocess_official(raw)
        if self.strict_prefix:
            out = await self._score_no_error(code, mode="complete")
            return out
        terminal = any(marker in raw for marker in _TERMINAL_MARKERS)
        if not code.strip():
            return float("-inf") if terminal else 0.0
        if terminal:
            # The solution can no longer change: apply the strict check.
            out = await self._score_no_error(code, mode="complete")
            return out
        if self._ec_head is not None and not self._ec_block_insert:
            # Inline insertion (e.g. `[insert].numpy()`): defer to complete().
            return 0.0
        for hung in self._hung_prefixes:
            if code.startswith(hung):
                return 0.0
        # Syntax is only meaningful for head + solution combined (the solution
        # may be an indented function body).
        combined = (self._ec_head or "") + code
        status = _prefix_syntax_status(combined)
        if status == "incomplete":
            self.last_was_syntax_error = False
            return 0.0
        if status == "broken" and "skip_plt_cmds" not in self.code_context:
            # Unfixable. Matplotlib harnesses go to the subprocess instead:
            # their solution-line filtering can repair the parse.
            self.last_was_syntax_error = True
            return float("-inf")
        out = await self._score_no_error(code, mode="prefix")
        return out

    async def complete(self, context: List[bytes]):
        # Apply transformation before processing
        if self.f is not None:
            context = self.f(context)
        code = _postprocess_official(self._bytes_to_str(context))
        # Empty completions never define `result`; skip the subprocess.
        if not code:
            return float("-inf")
        out = await self._score_no_error(code, mode="complete")
        return out

    def _complete_script(self, complete_code: str, ok: str, bad: str, syntax: str) -> str:
        """Harness script for complete solutions: run test_execution as-is."""
        return textwrap.dedent(
            f"""
            import sys, warnings, os, ast
            warnings.filterwarnings("ignore")
            os.environ.setdefault("PYTHONWARNINGS", "ignore")
            os.environ.setdefault("MPLBACKEND", "Agg")

            OK, BAD, SYNTAX = {ok!r}, {bad!r}, {syntax!r}
            code_context = {self.code_context!r}
            answer = {complete_code!r}
            try:
                g = {{}}
                exec(code_context, g, g)
                te = g.get("test_execution")
                if not callable(te):
                    print(BAD); raise SystemExit(0)
                # Syntax-check the answer as the harness runs it: inserted
                # into exec_context, matplotlib line filtering applied.
                to_check = answer
                skip = g.get("skip_plt_cmds")
                if callable(skip):
                    to_check = "\\n".join(filter(skip, to_check.split("\\n")))
                exec_context = g.get("exec_context")
                if isinstance(exec_context, str) and "[insert]" in exec_context:
                    to_check = exec_context.replace("[insert]", to_check)
                try:
                    ast.parse(to_check, filename="<answer>", mode="exec")
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

    def _prefix_script(self, prefix_code: str, ok: str, bad: str, syntax: str) -> str:
        """
        Stateless prefix script: exec the code_context to build the test
        inputs, then delegate to the shared `_judge_prefix`.
        """
        return (
            _PREFIX_JUDGE_SRC
            + textwrap.dedent(
                f"""
            import warnings, os
            warnings.filterwarnings("ignore")
            os.environ.setdefault("PYTHONWARNINGS", "ignore")
            os.environ.setdefault("MPLBACKEND", "Agg")

            OK, BAD, SYNTAX = {ok!r}, {bad!r}, {syntax!r}
            code_context = {self.code_context!r}
            answer = {prefix_code!r}
            head = {self._ec_head!r}
            n_cases = {self._n_test_cases}
            try:
                g = {{}}
                exec(code_context, g, g)
                gtc = g.get("generate_test_case")
                if head is None or not callable(gtc):
                    # Unknown harness structure: never kill a prefix we cannot
                    # faithfully execute.
                    print(OK); raise SystemExit(0)
                try:
                    test_inputs = [gtc(i + 1)[0] for i in range(n_cases)]
                except Exception:
                    # Test-input generation is harness-side: defer.
                    print(OK); raise SystemExit(0)
                print(_judge_prefix(
                    answer, head, g.get("skip_plt_cmds"), test_inputs, OK, BAD, SYNTAX))
            except SystemExit:
                raise
            except Exception:
                print(BAD)
            """
            )
        ).strip()

    async def _run_prefix_script(self, prefix_code, fallback_script, ok, bad, syntax):
        """
        Run a prefix check, preferring a warm per-task session (task setup
        executed once, each check forks from it). Any session/backend failure
        falls back to the stateless paths.
        """
        if (
            self.use_forkserver
            and self._ec_head is not None
            and self._ec_block_insert
            and os.environ.get("DS1000_FORKSERVER_SESSIONS", "1") != "0"
        ):
            try:
                executor = shared_executor(self.python_executable, self.extra_env)
                return await executor.run_session(
                    skey=self._session_key,
                    setup=self._session_setup_script(),
                    body=self._session_body_script(prefix_code, ok, bad, syntax),
                    fallback=fallback_script,
                    timeout=self.timeout_seconds,
                )
            except ForkserverUnavailable:
                # Keep the chain interceptable for subclasses overriding
                # _run_script (it falls back to the subprocess itself).
                return await self._run_script(fallback_script)
        return await self._run_script(fallback_script)

    def _session_setup_script(self) -> str:
        """
        Once-per-task session setup: define the shared `_judge_prefix`, exec
        the code_context, and build `head`/`skip`/`test_inputs` for body forks.
        """
        return (
            _PREFIX_JUDGE_SRC
            + textwrap.dedent(
                f"""
            import warnings, os
            warnings.filterwarnings("ignore")
            os.environ.setdefault("PYTHONWARNINGS", "ignore")
            os.environ.setdefault("MPLBACKEND", "Agg")

            code_context = {self.code_context!r}
            head = {self._ec_head!r}
            n_cases = {self._n_test_cases}
            g = {{}}
            exec(code_context, g, g)
            gtc = g.get("generate_test_case")
            if not callable(gtc):
                raise RuntimeError("unknown harness structure")
            skip = g.get("skip_plt_cmds")
            test_inputs = [gtc(i + 1)[0] for i in range(n_cases)]
            # Warm the head's imports so per-check forks hit sys.modules.
            for _line in head.split(chr(10)):
                if _line.startswith(("import ", "from ")):
                    try:
                        exec(_line, {{}}, {{}})
                    except Exception:
                        pass
            """
            )
        ).strip()

    def _session_body_script(self, prefix_code: str, ok, bad, syntax) -> str:
        """Per-check body, forked from the session: delegate to `_judge_prefix`."""
        return textwrap.dedent(
            f"""
            OK, BAD, SYNTAX = {ok!r}, {bad!r}, {syntax!r}
            print(_judge_prefix({prefix_code!r}, head, skip, test_inputs, OK, BAD, SYNTAX))
            """
        ).strip()

    async def _run_script(self, script: str):
        """
        Run a harness script; return its combined stdout+stderr, or None on
        timeout. Uses the fork-server when enabled, falling back to a fresh
        sandboxed subprocess on any backend failure.
        """
        if self.use_forkserver:
            try:
                executor = shared_executor(self.python_executable, self.extra_env)
                return await executor.run(script, self.timeout_seconds)
            except ForkserverUnavailable:
                pass
        return await self._run_script_subprocess(script)

    async def _run_script_subprocess(self, script: str):
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

                # Hold a slot for the subprocess's whole lifetime so the child
                # watcher's per-process threads stay bounded under load.
                async with _subprocess_sem():
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            self.python_executable,
                            "-B",
                            path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                            cwd=td,
                        )
                    except (OSError, RuntimeError):
                        # fork/pid/thread exhaustion under load ("can't start
                        # new thread" is a RuntimeError): transient, defer.
                        return None
                    try:
                        stdout_b, stderr_b = await asyncio.wait_for(
                            proc.communicate(), timeout=self.timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.communicate()
                        return None
        except subprocess.TimeoutExpired:
            return None

        return stdout_b.decode("utf-8", errors="replace") + stderr_b.decode(
            "utf-8", errors="replace"
        )

    async def _score_no_error(self, complete_code: str, mode: str = "complete") -> float:
        """
        Run the harness script (fork-server or subprocess): 0.0 if no error
        (incl. AssertionError), -inf otherwise. mode="complete" runs the full
        test_execution harness; "prefix" only the context head + solution.
        """
        cache_key = (mode, complete_code)
        cached = self._score_cache.get(cache_key)
        if cached is not None:
            self._score_cache.move_to_end(cache_key)
            self.cache_hits += 1
            value, syntax_error = cached
            self.last_was_syntax_error = syntax_error
            return value
        self.cache_misses += 1

        # Per-invocation nonce so solution code cannot spoof the verdict by
        # printing a guessable marker string.
        nonce = uuid.uuid4().hex[:12]
        OK = f"<<<OK:{nonce}>>>"
        BAD = f"<<<BAD:{nonce}>>>"
        SYNTAX = f"<<<SYNTAX:{nonce}>>>"

        if mode == "prefix":
            script = self._prefix_script(complete_code, OK, BAD, SYNTAX)
            out = await self._run_prefix_script(complete_code, script, OK, BAD, SYNTAX)
        else:
            script = self._complete_script(complete_code, OK, BAD, SYNTAX)
            out = await self._run_script(script)
        if out is None:
            # Timeout: a slow-but-valid prefix may still have a correct
            # continuation, so defer; only complete() treats it as fatal.
            self.last_was_syntax_error = False
            if mode == "prefix":
                self._hung_prefixes.append(complete_code)
                if len(self._hung_prefixes) > self._hung_prefixes_maxsize:
                    self._hung_prefixes.pop(0)
                return 0.0
            return float("-inf")

        ok = any(line.strip() == OK for line in out.splitlines())
        bad = any(line.strip() == BAD for line in out.splitlines())
        syntax = any(line.strip() == SYNTAX for line in out.splitlines())

        if mode == "prefix" and not (ok or bad or syntax):
            # No verdict marker (hard crash / garbled output): transient for a
            # prefix, like a timeout -- defer, uncached.
            self.last_was_syntax_error = False
            return 0.0

        self.last_was_syntax_error = bool(syntax)
        bad = bad or syntax
        value = 0.0 if ok and not bad else float("-inf")
        self._score_cache[cache_key] = (value, self.last_was_syntax_error)
        self._score_cache.move_to_end(cache_key)
        if len(self._score_cache) > self._score_cache_maxsize:
            self._score_cache.popitem(last=False)
        return value
