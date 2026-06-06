"""LCB prompt formatting + code extraction, faithful to official ``lcb_runner``.

Matches ``lcb_runner/prompts/code_generation.py`` (``get_generic_question_template_answer``
+ ``SYSTEM_MESSAGE_GENERIC``) and ``lcb_runner/utils/extraction_utils.extract_code``
(generic/chat style) so numbers are leaderboard-comparable. NOTE: official *base*-model
eval uses few-shot examples; the non-chat path here is zero-shot (system + template),
a documented deviation for base models — the chat path is exact.
"""
from __future__ import annotations

from typing import Mapping

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)

# Verbatim from lcb_runner PromptConstants.
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
    """Official get_generic_question_template_answer assembly."""
    body = f"### Question:\n{question_content}\n\n"
    if starter_code:
        body += f"### Format: {_FORMATTING_WITH_STARTER}\n"
        body += f"```python\n{starter_code}\n```\n\n"
    else:
        body += f"### Format: {_FORMATTING_WITHOUT_STARTER}\n"
        body += "```python\n# YOUR CODE HERE\n```\n\n"
    body += "### Answer: (use the provided format with backticks)\n\n"
    return body


def format_lcb_prompt(row: Mapping[str, str], tokenizer=None,
                      chat_template: bool = False) -> str:
    """Full prompt string from a snapshot row (needs question_content, starter_code).

    chat_template=True: system + user template via the model chat template (exact
    official chat protocol). Else: SYSTEM_MESSAGE + template as a completion string."""
    body = _user_body(row.get("question_content", ""), row.get("starter_code", "") or "")
    if chat_template and tokenizer is not None:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": body}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_MESSAGE}\n\n{body}"


def extract_code(model_output: str) -> str:
    """Code between the last two ``` fences; "" if there are fewer than two.

    Matches lcb_runner extraction_utils.extract_code (generic/chat style)."""
    lines = model_output.split("\n")
    fence_idxs = [i for i, ln in enumerate(lines) if "```" in ln]
    if len(fence_idxs) < 2:
        return ""
    return "\n".join(lines[fence_idxs[-2] + 1: fence_idxs[-1]])
