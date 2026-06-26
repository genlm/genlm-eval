"""Prompt formatting + code extraction kept identical to the official ``lcb_runner``
so numbers are leaderboard-comparable (base models: style="genericbase" = the official
few-shot protocol, paired with whole-output extraction). NB lcb_runner uses the
Meta-Llama-3-8B-Instruct chat template for all Llama-3.x instruct models."""
from __future__ import annotations

import json
from pathlib import Path
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
# lcb_runner DeepSeekCodeInstruct style (raw ### Instruction/Response completion string).
SYSTEM_MESSAGE_DEEPSEEK = (
    "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed "
    "by DeepSeek Company, and you answer questions related to computer science."
)
DEFAULT_STOP = ["###"]  # lcb_runner default --stop; apply at decode time (see cookbook)
STYLES = ("generic", "codeqwen", "deepseek", "genericbase")
RAW_STYLES = ("codeqwen", "deepseek", "genericbase")  # raw strings (no apply_chat_template)

# lcb_runner GenericBase few-shot examples (vendored verbatim); official uses only [0].
_FEWSHOT_DIR = Path(__file__).parent / "vendored" / "few_shot_examples" / "generation"
_FEWSHOT: dict = {}


def _fewshot_example(kind: str) -> dict:
    if kind not in _FEWSHOT:
        _FEWSHOT[kind] = json.loads((_FEWSHOT_DIR / f"{kind}.json").read_text())[0]
    return _FEWSHOT[kind]

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


def _deepseek_body(question_content: str, starter_code: str) -> str:
    """Official get_deepseekcode_question_template_answer (### Instruction/Response)."""
    body = ("### Instruction: You will be given a question (problem specification) and "
            "will generate a correct Python program that matches the specification and "
            "passes all tests. You will NOT return anything except for the program.\n\n")
    body += f"Question:\n{question_content}\n\n"
    if starter_code:
        body += f"### Instruction: {_FORMATTING_WITH_STARTER}\n```python\n{starter_code}\n```\n\n"
    else:
        body += f"### Instruction: {_FORMATTING_WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    body += "### Response:\n\n"
    return body


def _genericbase_body(question_content: str, starter_code: str) -> str:
    """Official get_base_model_question_template_answer (one few-shot example, no system msg)."""
    example = _fewshot_example("func" if starter_code else "stdin")

    def fmt(question, sample_code, answer):
        out = f"### Question\n{question}\n\n"
        if starter_code:  # official keys this on the REAL question's starter_code for both
            out += f"### Starter Code\n{sample_code}\n\n"
        out += f"### Answer\n\n{answer}"
        if answer:
            out += "\n\n"
        return out

    return (fmt(example["question"], example.get("sample_code", ""), example["answer"])
            + fmt(question_content, starter_code, ""))


def format_lcb_prompt(row: Mapping[str, str], tokenizer=None,
                      chat_template: bool = False, style: str = "generic",
                      enable_thinking: bool | None = None) -> str:
    """Prompt for an lcb_runner LMStyle: "generic" (LLaMa3, via chat template when
    chat_template=True), "codeqwen" (CodeQwenInstruct, raw <|im_*|> string), or
    "deepseek" (DeepSeekCodeInstruct, raw ### Instruction/Response string).

    enable_thinking forwards to apply_chat_template (Qwen3-style reasoning toggle); left
    out of the call when None so non-reasoning templates are unaffected."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}; got {style!r}")
    qc, sc = row.get("question_content", ""), row.get("starter_code", "") or ""
    if style == "codeqwen":
        # official joins system + body with a blank line ("...<|im_start|>user\n\n...")
        return f"{SYSTEM_MESSAGE_CODEQWEN}\n\n{_codeqwen_body(qc, sc)}"
    if style == "deepseek":
        return f"{SYSTEM_MESSAGE_DEEPSEEK}\n\n{_deepseek_body(qc, sc)}"
    if style == "genericbase":
        return _genericbase_body(qc, sc)
    body = _user_body(qc, sc)
    if chat_template and tokenizer is not None:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": body}]
        kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    return f"{SYSTEM_MESSAGE}\n\n{body}"


def extract_code(model_output: str, style: str = "generic") -> str:
    """Code between the last two ``` fences (last block if 3+); "" if fewer than two.
    style="genericbase" = whole stripped output. Matches lcb_runner extract_code."""
    # Reasoning models (Qwen3, R1, ...) emit <think>...</think> before the answer; keep only the
    # post-think answer so a code fence inside the reasoning can't be mistaken for the solution.
    # No </think> (every existing non-reasoning model) leaves the output unchanged.
    if "</think>" in model_output:
        model_output = model_output.rsplit("</think>", 1)[1]
    if style == "genericbase":
        return model_output.strip()
    lines = model_output.split("\n")
    fence_idxs = [i for i, ln in enumerate(lines) if "```" in ln]
    if len(fence_idxs) < 2:
        return ""
    return "\n".join(lines[fence_idxs[-2] + 1: fence_idxs[-1]])


def extract_code_prefix(model_output: str, style: str = "generic") -> str:
    """Code being written, for prefix scoring: text after the last open fence, or
    "" when no block is open. Deferring on a closed block (a later block could
    supersede it) keeps prefix consistent with ``extract_code`` at complete.
    style="genericbase" = whole stripped output."""
    if style == "genericbase":
        return model_output.strip()
    lines = model_output.split("\n")
    fence_idxs = [i for i, ln in enumerate(lines) if "```" in ln]
    if len(fence_idxs) % 2 == 1:  # block open: judge text after the last fence
        return "\n".join(lines[fence_idxs[-1] + 1:])
    return ""  # closed or no block: defer to complete()


def decode_context(context) -> str:
    """Decode a genlm.control context (str/bytes/list of byte tokens or int byte
    ids) into text."""
    if not context:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, bytes):
        return context.decode("utf-8", errors="ignore")
    pieces = []
    for tok in context:
        if isinstance(tok, int):
            pieces.append(bytes([tok]))
        elif isinstance(tok, bytes):
            pieces.append(tok)
        else:
            pieces.append(str(tok).encode("utf-8", errors="ignore"))
    return b"".join(pieces).decode("utf-8", errors="ignore")
