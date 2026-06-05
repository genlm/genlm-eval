"""Multiprocessing wrapper around the vendored LCB ``run_test``.

``run_test`` calls ``reliability_guard()`` (destructive interpreter patching) and
uses ``signal.alarm`` (main-thread only), so it must run in a forked child. This
mirrors official LCB ``compute_code_generation_metrics.check_correctness``.

Ported verbatim from the genlm/latent PR (jac/add-livecodebench), only the import
path changes (vendored util now lives in this domain).
"""
from __future__ import annotations

import json
import multiprocessing
from typing import Any, Dict, List, Tuple

from genlm.eval.domains.livecodebench._vendor_testing_util import run_test


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample: Dict[str, str], generation: str,
                      timeout: float = 6.0, debug: bool = False
                      ) -> Tuple[List[Any], Dict[str, Any]]:
    """Run ``generation`` against ``sample['input_output']`` in a forked child.

    Returns ``(results, metadata)`` where ``results`` is a per-test list of
    truthy/falsy outcomes (``True``/``False`` per test, or sentinel ints like
    ``-1``/``-2``/``-4`` on global failure)."""
    n_tests = len(json.loads(sample["input_output"])["inputs"])
    # The vendored run_test passes ``timeout`` to ``signal.alarm``, which requires
    # an int, so coerce here (our callers default to a float like 6.0).
    run_timeout = max(1, int(timeout))
    manager = multiprocessing.Manager()
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
    if not result:
        return [-1] * n_tests, {"error": "global timeout"}
    return list(result[0]), (dict(metadata_list[0]) if metadata_list else {})


def passed_all(sample: Dict[str, str], generation: str, timeout: float = 6.0) -> bool:
    """Strict 0/1: True iff every test ran and returned exactly pass (``== 1``).

    Sentinels (``-1``/``-2``/``-4``), ``False``, and an empty list all fail."""
    results, _ = check_correctness(sample, generation, timeout=timeout)
    return bool(results) and all(r == 1 for r in results)
