"""Differential grading against the unmodified upstream Multi-LCB executor.

The strongest replication check: for every shared language and every solution variant, our
vendored executor and the original ``Multi-LCB/lcb_runner/evaluation/testing_plang.py`` must
return the same solved/not-solved verdict. The only behavioral edits we made that could change
a verdict are (a) the memory-cap exemption, which we extended only to julia (absent from the
upstream eval_scripts, so no shared language is affected), and (b) the exit-code-first status
fix, which only matters when a correct program prints a status keyword to stderr (not the case
for any standard variant here). So agreement is expected across the board.

Opt-in: marked ``differential`` and skipped unless ../Multi-LCB is importable and the toolchain
is present. Run with the mlcb-tools env on PATH; deselect elsewhere with ``-m "not differential"``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from fixtures.mlcb_solutions import SOLUTIONS, SUM_N_INPUTS, SUM_N_OUTPUTS

from genlm.eval.domains.livecodebench_multilingual import (
    LocalSubprocessExecutor,
    is_toolchain_available,
)

_MULTI_LCB = Path(__file__).resolve().parents[1].parent / "Multi-LCB"


def _load_upstream():
    """Import the upstream testing_plang as a standalone module, or None if unavailable."""
    path = _MULTI_LCB / "lcb_runner" / "evaluation" / "testing_plang.py"
    if not path.exists():
        return None
    root = str(_MULTI_LCB)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        spec = importlib.util.spec_from_file_location("upstream_testing_plang", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 - any upstream import failure means skip the differential
        return None


_UPSTREAM = _load_upstream()
_SHARED = sorted(set(SOLUTIONS) & set(_UPSTREAM.eval_scripts)) if _UPSTREAM else []
_CASES = [(lang, v) for lang in _SHARED for v in SOLUTIONS[lang]]


def _upstream_solved(code, language):
    scores, _ = _UPSTREAM.eval_plang_code(
        code, list(SUM_N_INPUTS), list(SUM_N_OUTPUTS), language, 20
    )
    return bool(scores) and all(s.value > 0 for s in scores)


@pytest.mark.skipif(not _MULTI_LCB.exists(), reason="../Multi-LCB not present")
def test_upstream_actually_loaded():
    # Guard against silent self-disable: if the repo is present, the import must have succeeded
    # and produced cases. Otherwise a swallowed import error looks identical to "upstream absent".
    assert _UPSTREAM is not None, (
        "../Multi-LCB present but testing_plang failed to import"
    )
    assert _CASES, "upstream loaded but no shared languages resolved"


@pytest.mark.differential
@pytest.mark.skipif(_UPSTREAM is None, reason="../Multi-LCB not importable")
@pytest.mark.parametrize("language,variant", _CASES)
def test_verdict_agrees_with_upstream(language, variant):
    if not is_toolchain_available(language):
        pytest.skip(f"{language} toolchain absent")
    code = SOLUTIONS[language][variant]
    ex = LocalSubprocessExecutor()  # lenient, matching upstream defaults
    ex.prepare(language)
    ours, _ = ex.run(code, SUM_N_INPUTS, SUM_N_OUTPUTS, language, 20.0)
    theirs = _upstream_solved(code, language)
    assert ours == theirs, f"{language}/{variant}: ours={ours} upstream={theirs}"
