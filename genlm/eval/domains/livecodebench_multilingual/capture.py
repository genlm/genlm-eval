"""Full per-test execution capture for the multilingual-LCB (Agnostics) executor.

The graded path short-circuits on the first failing test and keeps only a solved/unsolved
verdict. `capture_run` re-runs the loop without short-circuiting and records every test's stdout,
stderr, exit code, and verdict. It reuses the vendored primitives and comparators, so its
per-test verdicts and aggregate `solved` match the official grader (checked for parity); the
vendored file is untouched.

For OCaml it compiles once (ocamlopt, ocamlc fallback) and runs the binary per test rather than
letting `ocaml Main.ml` recompile every invocation; a compile failure maps to EXECFAIL for all
tests, as the official path does.
"""

from __future__ import annotations

import tempfile
import time
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

from .vendored.testing_plang import (
    SubprocessConfig,
    Status,
    TestScore,
    eval_scripts,
    get_build_status,
    get_run_status,
    match_tests_exact,
    match_tests_groud_truth,
    patch_prog,
    run,
)

# Languages run by interpreting the source each invocation (no compile step). `{path}` is the source.
_INTERPRETED: Dict[str, List[str]] = {
    "lua": ["luajit", "{path}"],
    "julia": ["julia", "--startup-file=no", "-O0", "{path}"],
    "r": ["Rscript", "{path}"],
}
# Languages compiled once then run per test. Each entry: list of (compile_args, run_args) to try
# in order; the first that builds a runnable binary wins. `{path}` is the source, `{exe}` the binary.
# OCaml: ocamlopt (native) first, ocamlc (bytecode) fallback; both use the stdlib only, matching
# `ocaml Main.ml` module resolution (a program needing Str/Unix fails under both, giving EXECFAIL).
_COMPILED: Dict[str, List[Tuple[List[str], List[str]]]] = {
    "ocaml": [
        (["ocamlopt", "{path}", "-o", "{exe}"], ["{exe}"]),
        (["ocamlc", "{path}", "-o", "{exe}"], ["{exe}"]),
    ],
    "fortran": [(["gfortran", "{path}", "-o", "{exe}"], ["{exe}"])],
}


# Per-test error codes, the genlm-rollouts executions convention (see livecodebench/capture.py).
# eval and rollouts share these integer values by contract, not by import.
OK, WRONG, TLE, RTE, COMPILE, HARNESS = 1, -2, -3, -4, -5, -1

_COMPILE_FAULTS = {
    Status.BuildFailed,
    Status.BuildTimeOut,
    Status.SyntaxError,
    Status.EmptyCode,
}
_RUNTIME_FAULTS = {
    Status.AbnormalTermination,
    Status.Exception,
    Status.ValueError,
    Status.OutOfMemory,
}


def _error_code(status, passed: bool) -> int:
    """Per-test outcome to rollouts error_code: 1 ok / -2 wrong / -3 timeout / -4 runtime /
    -5 compile / -1 harness."""
    if passed:
        return OK
    if status == Status.Done:
        return WRONG  # ran cleanly, output mismatch
    if status == Status.TimeoutExpired:
        return TLE
    if status in _COMPILE_FAULTS:
        return COMPILE
    if status in _RUNTIME_FAULTS:
        return RTE
    return HARNESS  # wall-cap ("capped") and anything unclassified


def _rec(test_idx, passed, output, error_message, status, time_s):
    return {
        "test_idx": int(test_idx),
        "passed": bool(passed),
        "output": output or "",
        "error_message": error_message or "",
        "error_code": _error_code(status, passed),
        "status": str(status),
        "time_s": float(time_s),
    }


def _verdict(out: str, inp: str, exp: str, grading: str) -> bool:
    if grading == "exact":
        scores, _ = match_tests_exact([out], [exp])
    else:
        scores, _ = match_tests_groud_truth([out], [inp], [exp])
    return scores[0] == TestScore.PASSED


def _build(language, path, exe, sconf):
    """Compile a compiled-language source once. Returns (run_args, None) on success or
    (None, build_stderr) if every recipe fails."""
    last_err = ""
    for comp_tmpl, run_tmpl in _COMPILED[language]:
        cargs = [a.format(path=str(path), exe=exe) for a in comp_tmpl]
        cres = run(cargs, timeout_seconds=sconf.build_timeout, sconf=sconf)
        if get_build_status(cres) == Status.BuildDone and Path(exe).exists():
            return [a.format(path=str(path), exe=exe) for a in run_tmpl], None
        last_err = cres.stderr or cres.stdout
    return None, last_err


def capture_run(
    code: str,
    inputs: List[str],
    outputs: List[str],
    language: str,
    timeout: float = 10.0,
    grading: str = "exact",
    max_completion_seconds: float = 1e9,
) -> Tuple[bool, List[dict]]:
    """Run every test of one solution and return (solved, per_test_records), no short-circuit.

    Each record has test_idx, passed, output (untruncated stdout), error_message, error_code
    (rollouts convention), status, and time_s. `solved` is all(per-test passed). A per-completion
    wall cap (max_completion_seconds) records tests past the cap as not-passed, status "capped".
    """
    n = len(outputs)
    if not (code or "").strip():
        recs = [
            _rec(
                i, False, "", "Empty string instead of a program", Status.EmptyCode, 0.0
            )
            for i in range(n)
        ]
        return False, recs

    code = patch_prog(code, language)
    ext = eval_scripts[language][1]
    recs: List[dict] = []
    t_start = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "Main" + ext)
        with open(path, "w", encoding="utf8") as f:
            f.write(code)
            f.flush()
        sconf = SubprocessConfig(plang=language, run_timeout=max(1, int(ceil(timeout))))
        sconf.set_cwd(tmp)

        if language in _COMPILED:
            exe = str(path.with_suffix(".exe"))
            run_args, build_err = _build(language, path, exe, sconf)
            if run_args is None:
                return False, [
                    _rec(i, False, "", build_err, Status.BuildFailed, 0.0)
                    for i in range(n)
                ]
        elif language in _INTERPRETED:
            run_args = [a.format(path=str(path)) for a in _INTERPRETED[language]]
        else:
            raise NotImplementedError(
                f"capture_run has no recipe for language {language!r}"
            )

        for idx, (inp, exp) in enumerate(zip(inputs, outputs)):
            if time.time() - t_start > max_completion_seconds:
                recs.append(
                    _rec(idx, False, "", "completion wall cap reached", "capped", 0.0)
                )
                continue
            t0 = time.time()
            res = run(
                run_args, input_data=inp, timeout_seconds=sconf.run_timeout, sconf=sconf
            )
            dt = time.time() - t0
            out = (res.stdout or "").strip()
            rs = get_run_status(res)
            if rs != Status.Done:
                recs.append(_rec(idx, False, out, res.stderr, rs, dt))
            else:
                passed = _verdict(out, inp if inp is not None else "", exp, grading)
                recs.append(
                    _rec(idx, passed, out, "" if passed else res.stderr, rs, dt)
                )

    solved = bool(recs) and all(r["passed"] for r in recs)
    return solved, recs
