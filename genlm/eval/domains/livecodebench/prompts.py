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
# lcb_runner CodeQwenInstruct style (manual chat tokens, NOT apply_chat_template).
SYSTEM_MESSAGE_CODEQWEN = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
)
DEFAULT_STOP = ["###"]  # lcb_runner default --stop

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
    """Full prompt string from a snapshot row, matching an lcb_runner LMStyle.

    style="generic" (LLaMa3 etc.): SYSTEM_MESSAGE + ### Question/Format/Answer template,
    via the model chat template when chat_template=True. style="codeqwen"
    (CodeQwenInstruct): the model-specific "helpful assistant" + manual <|im_*|> prompt
    (a raw completion string, no apply_chat_template)."""
    qc, sc = row.get("question_content", ""), row.get("starter_code", "") or ""
    if style == "codeqwen":
        return f"{SYSTEM_MESSAGE_CODEQWEN}\n{_codeqwen_body(qc, sc)}"
    body = _user_body(qc, sc)
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
