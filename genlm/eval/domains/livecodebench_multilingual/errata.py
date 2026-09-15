"""Problems that cannot be scored by the harness.

Three categories: the problem accepts several correct answers and the harness compares
against one scraped answer, the problem needs an interactive judge, or the shipped test data
is wrong.

``ERRATA_UPSTREAM`` holds the ids LiveCodeBench records in its own ``ERRATA.md``. The harness
does not read that file, so the problems stay in every loaded dataset.

``ERRATA_VERIFIED`` holds the ids established here by grading a verified reference solution
against the official test data.

Each entry is a property of the problem statement and its test data, so it applies to every
target language. A given slice contains only part of this list: some upstream ids are LeetCode
problems, which the stdin filter removes, and some predate the default date window.

Evidence per problem, and the reference solutions behind ``ERRATA_VERIFIED``:
https://huggingface.co/datasets/samuki-hf/ocaml-reference-solutions
"""

from typing import Any, Iterable, List, Mapping

# Verbatim from LiveCodeBench's ERRATA.md.
ERRATA_UPSTREAM = {
    "abc311_c": "multiple-solutions",
    "abc326_d": "multiple-solutions",
    "abc327_b": "multiple-solutions",
    "abc333_e": "multiple-solutions",
    "abc343_a": "multiple-solutions",
    "abc343_e": "multiple-solutions",
    "abc362_c": "multiple-solutions",
    "arc185_c": "multiple-solutions",
    "find-words-containing-character": "multiple-solutions",
    "find-the-peaks": "multiple-solutions",
    "generate-binary-strings-without-adjacent-zeros": "multiple-solutions",
    # The submission must interact with a judge the harness does not provide.
    "abc337_e": "interactive",
    "abc355_e": "interactive",
    # The shipped test data is wrong.
    "abc350_c": "erroneous-tests",
    "arc189_a": "erroneous-tests",
    "apply-operations-to-make-string-empty": "erroneous-tests",
    "most-frequent-ids": "erroneous-tests",
}

# Established here by grading a verified reference against the official tests.
ERRATA_VERIFIED = {
    "abc363_f": "multiple-solutions",
    "abc366_g": "multiple-solutions",
    "abc373_g": "multiple-solutions",
    "abc396_e": "multiple-solutions",
    "abc397_d": "multiple-solutions",
    "arc181_c": "multiple-solutions",
    "arc183_d": "multiple-solutions",
    "arc188_c": "multiple-solutions",
    "arc190_a": "multiple-solutions",
    "arc191_c": "multiple-solutions",
    "arc195_c": "multiple-solutions",
    # A verified reference fails only on tests that contradict the problem's constraints
    # or state the wrong answer.
    "abc392_f": "erroneous-tests",
    "arc192_b": "erroneous-tests",
}

ERRATA = {**ERRATA_UPSTREAM, **ERRATA_VERIFIED}


def drop_errata(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Drop rows whose ``question_id`` appears on either list."""
    return [r for r in rows if str(r.get("question_id")) not in ERRATA]
