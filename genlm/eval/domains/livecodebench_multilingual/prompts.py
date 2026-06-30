"""Per-language prompt construction for multilingual LiveCodeBench (stdin/stdout problems).

Mirrors Multi-LCB's stdin prompt so the 12 Multi-LCB languages are prompt-comparable to the
paper. ``extract_code`` matches Multi-LCB's extractor (first fenced block, placeholder
stripped), not the Python-only domain's, so grading matches the Multi-LCB pipeline.
"""

import re
from typing import Dict, List

from .dataset import Language, resolve_language

# Verbatim from Multi-LCB PromptConstants (MIT), with the placeholders kept as {fields}.
SYSTEM_MESSAGE_GENERIC = (
    "You are an expert {display} programmer. You will be given a question "
    "(problem specification) and will generate a correct {display} program that "
    "matches the specification and passes all tests."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the {md_fence} program runs, it reads the inputs, "
    "runs the algorithm and writes output to STDOUT.\n\n"
)

# Code-block regex matching Multi-LCB extraction_utils.extract_code (MIT), extended with the
# 5 low-resource language tags so their fence is stripped too (no effect on the 12).
_CODE_BLOCK_RE = re.compile(
    r"```(\s*(python|java|c\+\+|cpp|csharp|c\#|ts|js|typescript|javascript|rust|ruby|go|"
    r"php|scala|kotlin|lua|julia|ocaml|fortran|r)\s*\n)?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
# Comment tokens for the placeholder line, covering all 17 languages' comment styles.
_PLACEHOLDER_RE = re.compile(
    r"(#+|//+|--+|!+|\(\*) YOUR CODE HERE", re.DOTALL | re.IGNORECASE
)


def extract_code(model_output) -> str:
    """First fenced code block, matching Multi-LCB's extractor.

    Drops a leading </think> span, takes the first ``` block, and strips the "YOUR CODE HERE"
    placeholder. The Python-only domain's extractor takes the last block and keeps the placeholder.
    """
    if not model_output:
        return ""
    t = model_output.find("</think>")
    if t >= 0:
        model_output = model_output[t + 8 :].strip()
    m = _CODE_BLOCK_RE.search(model_output)
    if not m:
        return ""
    return _PLACEHOLDER_RE.sub("", m.group(3))


def _system_message(lang: Language) -> str:
    msg = SYSTEM_MESSAGE_GENERIC.format(display=lang.display)
    # Low-resource languages get an extra guidance line; empty for the 12 Multi-LCB
    # languages, so their system message stays byte-identical to Multi-LCB's.
    if lang.prompt_nudge:
        msg = f"{msg}\n\n{lang.prompt_nudge}"
    return msg


def _user_body(question_content: str, lang: Language) -> str:
    """Multi-LCB get_enhanced_question_template_answer, no-starter (stdin) branch."""
    body = f"### Question:\n{question_content}\n\n"
    # Trailing newline matches Multi-LCB byte-for-byte (FORMATTING ends with "\n\n", plus one more).
    body += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE.format(md_fence=lang.md_fence)}\n"
    body += f"```{lang.md_fence}\n{lang.comment} YOUR CODE HERE\n```\n\n"
    body += "### Answer: (use the provided format with backticks)\n\n"
    return body


def multilingual_chat_messages(instance) -> List[Dict[str, str]]:
    """The [system, user] chat messages for ``instance`` (for chat/API model adapters)."""
    lang = resolve_language(instance.language)
    return [
        {"role": "system", "content": _system_message(lang)},
        {"role": "user", "content": _user_body(instance.question_content, lang)},
    ]


def agnostics_chat_messages(instance) -> List[Dict[str, str]]:
    """Agnostics Ag-LCB-X eval prompt: one user message naming the target language.

    Mirrors agnostics-framework make_prompt_from_lcbx_row (a "# Problem / # Task" block, no
    system message). Pair with grading="exact" for an Agnostics-parity run.
    """
    lang = resolve_language(instance.language)
    user = (
        f"# Problem\n{instance.question_content}\n\n"
        "# Task\nProvide a full implementation of the specified program in a Markdown code "
        f"block.\nUse the following programming language: {lang.key}\n"
    )
    return [{"role": "user", "content": user}]


def format_multilingual_prompt(
    tokenizer,
    instance,
    use_chat_format: bool = False,
    enable_thinking: bool | None = None,
) -> List[int]:
    """Build the multilingual LCB prompt for ``instance`` and return token ids.

    use_chat_format=True applies the tokenizer's chat template (instruct models); otherwise
    the system and user messages are concatenated as a raw completion string. Mirrors the
    existing ``default_prompt_formatter`` interface.
    """
    messages = multilingual_chat_messages(instance)
    if use_chat_format and tokenizer is not None:
        kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kw
        )
        # The chat template already inserts the BOS; avoid a second one on re-encode.
        return tokenizer.encode(text, add_special_tokens=False)
    text = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    return tokenizer.encode(text)


def chat_messages(instance) -> List[Dict[str, str]]:
    """Chat messages for ``instance`` in its source's prompt style: Multi-LCB languages get the
    Multi-LCB prompt, Agnostics low-resource languages get the Agnostics prompt with the
    per-language nudge. Prefer this over the style-specific builders so each prompt matches its
    dataset."""
    lang = resolve_language(instance.language)
    if lang.source == "agnostics":
        msgs = agnostics_chat_messages(instance)
        if lang.prompt_nudge:
            msgs = [{**msgs[0], "content": msgs[0]["content"] + "\n" + lang.prompt_nudge}]
        return msgs
    return multilingual_chat_messages(instance)


def default_grading(language) -> str:
    """Grading comparator matching each prompt source: ``exact`` (Agnostics rstrip-equality) for the
    Agnostics low-resource languages, ``lenient`` (Multi-LCB per-line comparator) otherwise."""
    return "exact" if resolve_language(language).source == "agnostics" else "lenient"


def format_prompt(
    tokenizer,
    instance,
    use_chat_format: bool = False,
    enable_thinking: bool | None = None,
) -> List[int]:
    """Source-correct token ids for ``instance`` (Multi-LCB or Agnostics prompt by language source).

    The generation-side analogue of ``format_multilingual_prompt`` but style-selecting via
    ``chat_messages``. ``enable_thinking=None`` omits the toggle for models without a thinking mode.
    """
    messages = chat_messages(instance)
    if use_chat_format and tokenizer is not None:
        kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kw
        )
        return tokenizer.encode(text, add_special_tokens=False)
    # raw-completion fallback: agnostics carries one (user) message, Multi-LCB two (system+user)
    text = "\n\n".join(m["content"] for m in messages)
    return tokenizer.encode(text)
