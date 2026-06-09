"""Prompt formatting + code extraction kept identical to the official ``lcb_runner``
so numbers are leaderboard-comparable. Only deviation: the non-chat path is
zero-shot (lcb_runner few-shots base models); the chat path is exact."""
from __future__ import annotations

from typing import Mapping

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)
# lcb_runner CodeQwenInstruct style (manual chat tokens, NOT apply_chat_template).
SYSTEM_MESSAGE_CODEQWEN = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
)
DEFAULT_STOP = ["###"]  # lcb_runner default --stop; apply at decode time (see cookbook)
STYLES = ("generic", "codeqwen")

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


def _codeqwen_body(question_content: str, starter_code: str) -> str:
    """Official get_codeqwen_question_template_answer (manual <|im_*|> tokens)."""
    body = ("You will be given a question (problem specification) and will generate a "
            "correct Python program that matches the specification and passes all tests. "
            "You will NOT return anything except for the program.\n\n")
    body += f"Question: {question_content}\n\n"
    if starter_code:
        body += f"{_FORMATTING_WITH_STARTER}\n```python\n{starter_code}\n```\n\n<|im_end|>\n"
    else:
        body += f"{_FORMATTING_WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    body += "<|im_start|>assistant\n"
    return body


def format_lcb_prompt(row: Mapping[str, str], tokenizer=None,
                      chat_template: bool = False, style: str = "generic") -> str:
    """Prompt for an lcb_runner LMStyle: "generic" (LLaMa3, via chat template when
    chat_template=True) or "codeqwen" (CodeQwenInstruct, raw <|im_*|> string)."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}; got {style!r}")
    qc, sc = row.get("question_content", ""), row.get("starter_code", "") or ""
    if style == "codeqwen":
        # official joins system + body with a blank line ("...<|im_start|>user\n\n...")
        return f"{SYSTEM_MESSAGE_CODEQWEN}\n\n{_codeqwen_body(qc, sc)}"
    body = _user_body(qc, sc)
    if chat_template and tokenizer is not None:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": body}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_MESSAGE}\n\n{body}"


def extract_code(model_output: str) -> str:
    """Code between the last two ``` fences (last block if 3+); "" if fewer than two.

    Matches lcb_runner extract_code (generic/chat style)."""
    lines = model_output.split("\n")
    fence_idxs = [i for i, ln in enumerate(lines) if "```" in ln]
    if len(fence_idxs) < 2:
        return ""
    return "\n".join(lines[fence_idxs[-2] + 1: fence_idxs[-1]])
