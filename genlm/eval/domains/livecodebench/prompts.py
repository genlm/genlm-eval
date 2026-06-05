"""LCB prompt formatting + code extraction.

Adapted from LiveCodeBench ``lcb_runner/prompts/code_generation.py`` and
``lcb_runner/utils/extraction_utils.py`` (the upstream Qwen formatter hardcodes a
private tokenizer path; this version takes the training tokenizer instead).
"""
from __future__ import annotations

from typing import Mapping, Optional

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)

_FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)
_FORMATTING_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm and writes output to STDOUT."
)


def _user_body(question_content: str, starter_code: str) -> str:
    body = (
        "You will be given a question (problem specification) and will generate a "
        "correct Python program that matches the specification and passes all "
        "tests. You will NOT return anything except for the program.\n\n"
    )
    body += f"Question:\n{question_content}\n\n"
    if starter_code:
        body += f"{_FORMATTING_WITH_STARTER}\n"
        body += f"```python\n{starter_code}\n```\n\n"
    else:
        body += f"{_FORMATTING_WITHOUT_STARTER}\n\n"
        body += "```python\n# YOUR CODE HERE\n```\n\n"
    return body


def format_lcb_prompt(row: Mapping[str, str], tokenizer=None, style: str = "qwen",
                      chat_template: bool = False) -> str:
    """Build the full prompt string from a snapshot row.

    ``row`` needs ``question_content`` and ``starter_code``. With
    ``chat_template=True`` and a ``tokenizer``, applies the model chat template;
    otherwise returns ``SYSTEM_MESSAGE`` + body as a plain string."""
    body = _user_body(row.get("question_content", ""), row.get("starter_code", "") or "")
    if chat_template and tokenizer is not None:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": body},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_MESSAGE}\n\n{body}"


def extract_code(model_output: str) -> str:
    """Return the last ``` fenced block; fall back to the stripped output.

    Mirrors LCB ``extract_code`` for the generic case (take the final fence pair)."""
    lines = model_output.split("\n")
    fence_idxs = [i for i, ln in enumerate(lines) if "```" in ln]
    if len(fence_idxs) < 2:
        return model_output.strip()
    return "\n".join(lines[fence_idxs[-2] + 1: fence_idxs[-1]])
