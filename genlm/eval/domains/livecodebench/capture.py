"""Full per-test execution capture for LiveCodeBench.
``run_test`` calls ``grade_call_based`` / ``grade_stdio`` as bare module-level
names (testing_util.py lines 476/497), so ``enable_capture()`` swaps in instrumented
copies that:

  * run every test (never short-circuit at the first wrong answer),
  * record each test's input / expected / actual output untruncated (no ``truncatefn``),
  * stop only on a hard fault (-3 timeout / -4 runtime error), where the interpreter
    state after a signal-driven exception is unsafe to keep running,
  * carry the per-test records back out through the returned ``metadata["executions"]``
    (which already flows to the parent via the harness pipe).

The official pass/fail is unchanged: ``passed_all`` is ``all(r > 0 for r in results)``.
"""
from __future__ import annotations

import faulthandler
import hashlib
import json
import signal
import time

from .vendored import testing_util as _tu
from .vendored.testing_util import (
    Capturing,
    call_method,
    clean_if_name,
    compile_code,
    convert_line_to_decimals,
    get_function,
    get_stripped_lines,
    import_string,
    make_function,
)

# Sentinel error codes (match the vendored harness convention).
OK, WRONG, TLE, RTE, COMPILE = 1, -2, -3, -4, -5


def _canon(x, kind):
    """Full, untruncated canonical string for a per-test value + its sha256.

    For call-based values JSON-encode so lists, tuples and nested structures hash stably;
    for stdio keep the raw stdout string.
    """
    if kind == "stdio":
        s = x if isinstance(x, str) else str(x)
    else:
        try:
            s = json.dumps(x, ensure_ascii=False, default=str, sort_keys=False)
        except Exception:
            s = repr(x)
    h = hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()
    return s, h


def _rec(test_idx, kind, inp, exp, out, passed, code, msg, t):
    # Store the output untruncated, plus hashes.
    _, eh = _canon(exp, kind)  # keep expected_hash for behavioral matching; drop the string
    so, oh = (_canon(out, kind) if out is not None else (None, None))
    return {
        "test_idx": test_idx,
        "output_kind": kind,
        "input": None,
        "expected": None,
        "expected_hash": eh,
        "output": so,
        "output_hash": oh,
        "passed": bool(passed),
        "error_code": int(code),
        "error_message": (None if msg is None else str(msg)),
        "time_s": round(float(t), 6),
    }


def grade_call_based_cap(code, all_inputs, all_outputs, fn_name, timeout):
    """Instrumented ``grade_call_based``: runs all tests, captures every return value."""
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return [COMPILE], {"executions": [], "n_tests": len(all_outputs), "error_code": COMPILE}
    method = get_function(compiled_sol, fn_name)
    if method is None:
        return [COMPILE], {"executions": [], "n_tests": len(all_outputs), "error_code": COMPILE}

    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    all_outputs = [json.loads(output) for output in all_outputs]

    results, records, total = [], [], 0.0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            dt = time.time() - start
            total += dt
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            ok = prediction == gt_out
            results.append(bool(ok))
            records.append(_rec(idx, "return", gt_inp, gt_out, prediction,
                                ok, OK if ok else WRONG, None if ok else "Wrong Answer", dt))
        except Exception as e:  # noqa: BLE001 (mirror vendored broad catch)
            signal.alarm(0)
            tle = "timeoutexception" in repr(e).lower()
            code_e = TLE if tle else RTE
            results.append(code_e)
            records.append(_rec(idx, "return", gt_inp, gt_out, None, False, code_e,
                                ("Time Limit Exceeded: " if tle else "Runtime Error: ") + repr(e), 0.0))
            faulthandler.disable()
            break  # interpreter state after a signal-driven exception is unsafe
        finally:
            signal.alarm(0)
            faulthandler.disable()

    return results, {"executions": records, "n_tests": len(all_outputs), "execution_time": total}


def grade_stdio_cap(code, all_inputs, all_outputs, timeout):
    """Instrumented ``grade_stdio``: runs all tests, captures every stdout untruncated."""
    code = clean_if_name(code)
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return [COMPILE], {"executions": [], "n_tests": len(all_outputs), "error_code": COMPILE}
    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return [COMPILE], {"executions": [], "n_tests": len(all_outputs), "error_code": COMPILE}

    results, records, total = [], [], 0.0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                dt = time.time() - start
                signal.alarm(0)
            except Exception as e:  # noqa: BLE001
                signal.alarm(0)
                tle = "timeoutexception" in repr(e).lower()
                code_e = TLE if tle else RTE
                results.append(code_e)
                # captured_output[0] is set on __exit__; read it after the with-block
                _err = ("Time Limit Exceeded: " if tle else "Runtime Error: ") + repr(e)
                faulthandler.disable()
                _broke = (idx, gt_inp, gt_out, code_e, _err)
                break
            finally:
                signal.alarm(0)
                faulthandler.disable()
        prediction = captured_output[0]
        ok, msg = _stdio_match(prediction, gt_out)
        results.append(bool(ok))
        records.append(_rec(idx, "stdio", gt_inp, gt_out, prediction,
                            ok, OK if ok else WRONG, None if ok else msg, dt))
    else:
        return results, {"executions": records, "n_tests": len(all_outputs), "execution_time": total}

    # loop broke on a hard fault: record it (stdout captured up to the fault)
    idx, gt_inp, gt_out, code_e, _err = _broke
    records.append(_rec(idx, "stdio", gt_inp, gt_out, (captured_output[0] if captured_output else ""),
                        False, code_e, _err, 0.0))
    return results, {"executions": records, "n_tests": len(all_outputs), "execution_time": total}


def _stdio_match(prediction, gt_out):
    """Replicate the vendored stdio comparison; returns (passed, message)."""
    sp = get_stripped_lines(prediction)
    sg = get_stripped_lines(gt_out)
    if len(sp) != len(sg):
        return False, "Wrong answer: mismatched output length"
    for i, (pl, gl) in enumerate(zip(sp, sg)):
        if pl == gl:
            continue
        okp, dp = convert_line_to_decimals(pl)
        okg, dg = convert_line_to_decimals(gl)
        if not okp or not okg or dp != dg:
            return False, f"Wrong answer at line {i}"
    return True, None


_ENABLED = False


def enable_capture():
    """Swap the vendored grade functions for the capturing copies (idempotent).

    Apply before forking the harness child so the patch is inherited by the child.
    """
    global _ENABLED
    # drift guard: the vendored originals must still be the functions we mirrored
    assert hasattr(_tu, "grade_call_based") and hasattr(_tu, "grade_stdio"), \
        "vendored testing_util changed: grade_call_based/grade_stdio missing"
    _tu.grade_call_based = grade_call_based_cap
    _tu.grade_stdio = grade_stdio_cap
    _ENABLED = True


def is_enabled():
    return _ENABLED
