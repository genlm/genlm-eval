"""Interop between the multilingual executor's verdicts and the genlm-rollouts failure taxonomy.

genlm-rollouts (the failure-analysis/report pipeline) buckets failing LiveCodeBench samples into
six categories and, in its per-test executions table, into six int8 error codes. Its own
classifier (``rollouts/analysis/lcb/classify_errors.py``) is hard-wired to the python-only
default-LCB ``run_test`` and cannot grade other languages. To fold a multilingual run into that
pipeline, map our executor ``status`` (a Multi-LCB ``Status`` enum name, e.g. ``"WrongAnswer"``,
``"TimeoutExpired"``, ``"OutOfMemory"``) onto the rollouts buckets. These two functions are that
mapping.

Notes on the lossy points (see MULTILINGUAL_LCB_PLAN.md):
- rollouts has no out-of-memory bucket, so ``OutOfMemory`` folds into ``runtime_error``; keep the
  raw status elsewhere (e.g. an ``exc_type`` column) if you need to recover it.
- A build/setup failure (compiler, ``npm i``, build timeout) maps to ``compile_error`` because the
  program never executed; a failure after execution started maps to ``runtime_error``.
- The per-test ``TestScore`` value must not be used as the error code: the executor collapses every
  exec failure to ``EXECFAIL`` (-5), which would alias rollouts' ``compile`` code (-5). Derive the
  error code from the ``status`` instead, which is what these functions do.
"""

from typing import Optional

# The genlm-rollouts failure taxonomy (rollouts/analysis/lcb/taxonomy.py CATEGORIES). Duplicated
# here as the contract; the consistency test asserts it still matches the rollouts repo.
ROLLOUTS_CATEGORIES = (
    "wrong_answer",
    "timeout",
    "runtime_error",
    "compile_error",
    "no_code",
    "harness_timeout",
)

# Statuses that mean the candidate did not fail, so there is no failure category. Mirrors the
# executor's Status.is_failure() (UNK/Done/BuildDone are non-failures), plus "ok", the string the
# executor emits on success. A test pins this alignment.
_PASS_STATUSES = frozenset({"ok", "UNK", "Done", "BuildDone"})

# Failing status name to rollouts failure category. Anything unrecognized falls back to runtime_error.
_CATEGORY = {
    "EmptyCode": "no_code",
    "WrongAnswer": "wrong_answer",
    "TimeoutExpired": "timeout",
    "BuildFailed": "compile_error",
    "BuildTimeOut": "compile_error",
    "SyntaxError": "compile_error",
    "NPMFailed": "compile_error",
    "Exception": "runtime_error",
    "AbnormalTermination": "runtime_error",
    "OutOfMemory": "runtime_error",
    "ValueError": "runtime_error",
    "UNK": "runtime_error",
}

# Per-test executions error code (rollouts exec_tables.py / capture.py):
# 1 pass, -2 wrong answer, -3 timeout, -4 runtime, -5 compile, -1 none/harness fault.
_ERROR_CODE = {
    "WrongAnswer": -2,
    "TimeoutExpired": -3,
    "Exception": -4,
    "AbnormalTermination": -4,
    "OutOfMemory": -4,
    "ValueError": -4,
    "UNK": -4,
    "BuildFailed": -5,
    "BuildTimeOut": -5,
    "SyntaxError": -5,
    "NPMFailed": -5,
    "EmptyCode": -1,
}


def rollouts_category(status: str) -> Optional[str]:
    """Map an executor status to a genlm-rollouts failure category.

    Returns ``None`` for a passing status (rollouts only categorizes failures), otherwise a member
    of :data:`ROLLOUTS_CATEGORIES`. Unknown statuses fall back to ``"runtime_error"``.
    """
    if status in _PASS_STATUSES:
        return None
    return _CATEGORY.get(status, "runtime_error")


def rollouts_error_code(status: str) -> int:
    """Map an executor status to a rollouts per-test executions error code (int8).

    Passing statuses map to ``1``; unknown statuses fall back to ``-4`` (runtime). Never returns the
    raw per-test ``EXECFAIL`` (-5) for a non-compile failure, so timeout/runtime stay distinct.
    """
    if status in _PASS_STATUSES:
        return 1
    return _ERROR_CODE.get(status, -4)
