"""Multiprocessing wrapper around the vendored LCB ``run_test``.

``run_test`` calls ``reliability_guard()`` (destructive interpreter patching) and
uses ``signal.alarm`` (main-thread only), so it must run in a forked child. This
mirrors official LCB ``compute_code_generation_metrics.check_correctness``.
"""
from __future__ import annotations

import json
import multiprocessing
from typing import Any, Dict, List, Tuple

from genlm.eval.domains.livecodebench.util.testing_util import run_test


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample: Dict[str, str], generation: str,
                      timeout: float = 6.0, debug: bool = False
                      ) -> Tuple[List[Any], Dict[str, Any]]:
    """Run ``generation`` against the tests in a forked child.

    ``results`` is per-test ``True``/``False`` (or sentinel ints ``-1``/``-2``/``-4``
    on failure)."""
    # Guard the input_output parse (official lcb_runner does this unguarded, but our
    # from_jsonl allows prompts-only snapshots where it may be absent/malformed): a bad
    # sample scores fail instead of crashing the whole eval run.
    try:
        n_tests = len(json.loads(sample["input_output"])["inputs"])
    except (KeyError, TypeError, ValueError):
        return [-1], {"error": "missing or malformed eval_sample"}
    run_timeout = max(1, int(timeout))  # signal.alarm needs an int
    # context manager shuts down the Manager's server process (else one leaks per call)
    with multiprocessing.Manager() as manager:
        result = manager.list()
        metadata_list = manager.list()
        p = multiprocessing.Process(
            target=_temp_run,
            args=(sample, generation, debug, result, metadata_list, run_timeout),
        )
        p.start()
        p.join(timeout=(timeout + 1) * n_tests + 5)
        if p.is_alive():
            p.kill()
            p.join()  # reap the killed child so it doesn't linger as a zombie
        if not result:
            return [-1] * n_tests, {"error": "global timeout"}
        return list(result[0]), (dict(metadata_list[0]) if metadata_list else {})


def passed_all(sample: Dict[str, str], generation: str, timeout: float = 6.0) -> bool:
    """True iff every test passed (``> 0``), matching official ``np.all(gen > 0)``."""
    if not sample or "input_output" not in sample:
        return False
    results, _ = check_correctness(sample, generation, timeout=timeout)
    return bool(results) and all(r > 0 for r in results)
