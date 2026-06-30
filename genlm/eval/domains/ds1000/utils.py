import ast
import os
import re


def _sandbox_env(td: str, extra_env: dict | None = None) -> dict:
    """
    Return sandbox env that keeps caches/configs inside td and avoids .pyc
    """
    base = dict(os.environ)
    base.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": td,
            "XDG_CACHE_HOME": os.path.join(td, "xdg_cache"),
            "XDG_CONFIG_HOME": os.path.join(td, "xdg_config"),
            "HF_HOME": os.path.join(td, "hf_home"),
            "TRANSFORMERS_CACHE": os.path.join(td, "hf_cache"),
            "TORCH_HOME": os.path.join(td, "torch_home"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONWARNINGS": "ignore",
        }
    )
    if extra_env:
        base.update(extra_env)
    return base


def _postprocess_official(t: str) -> str:
    # xlang-ai/DS-1000 postprocess: keep everything before the first fence. `# SOLUTION END`
    # is this project's generation marker (ds1000_common); base-model output never emits it,
    # so the cut is a no-op for leaderboard scoring.
    t = t.split("# SOLUTION END")[0]
    t = t.split("</code>")[0]
    t = t.replace("```python", "")
    t = t.split("```")[0]
    t = t.split("\nEND SOLUTION")[0]
    return t.replace("<code>", "")


def _strip_reasoning(t: str) -> str:
    # Reasoning models emit <think>...</think> before the answer; keep only the answer so
    # code fenced inside the reasoning is never mistaken for the solution.
    return t.rsplit("</think>", 1)[1] if "</think>" in t else t


def _postprocess_chat(t: str) -> str:
    # Chat/reasoning output puts the answer in a fenced ```python block after a prose (or
    # <think>) preamble. Take the last block of the answer; with no fence, fall back to
    # official (raw unchanged).
    # Do not dedent: a function-body answer is indented under its `def`, and stripping
    # that indent breaks the insertion.
    # Strip reasoning before the </code> cut: a </think> trace can itself contain a </code>
    # token, and cutting on that first would discard the real (post-</think>) answer and leave
    # a fenced block from the reasoning.
    t = _strip_reasoning(t).split("</code>")[0]
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", t, re.S)
    if blocks:
        t = blocks[-1]
    else:
        t = t.replace("```python", "")
        t = t.split("```")[0]
    t = t.split("# SOLUTION END")[0]
    t = t.split("\nEND SOLUTION")[0]
    return t.replace("<code>", "")


def _insert_slot_def(code_context: str) -> "str | None":
    """The `def NAME(...):` header on the line immediately preceding the `[insert]` slot in a
    DS-1000 code_context, or None when the slot is not a function body. DS-1000 insertion-type
    problems open `def f(df):` and expect the *indented body* at `[insert]`."""
    i = code_context.find("[insert]")
    if i < 0:
        return None
    for line in reversed(code_context[:i].splitlines()):
        if not line.strip():
            continue
        s = line.strip()
        return s if s.startswith("def ") and s.endswith(":") else None
    return None


def unwrap_redeclared_function(solution: str, code_context: str) -> str:
    """If the model answered a DS-1000 body-insertion slot with a *standalone* function that
    re-declares the one the slot already opens (`def f(df):` template + `def f(df): ...` answer),
    return just the function body so insertion doesn't `IndentationError`. Conservative: fires
    only when the slot is a `def`, the solution is one top-level FunctionDef, and the names match.
    Otherwise returns the solution unchanged, preserving a bare-body answer's indentation. See
    genlm/rollouts issue #18."""
    head = _insert_slot_def(code_context)
    if not head:
        return solution
    slot_name = head[4:head.find("(")].strip()
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return solution
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        fn = tree.body[0]
        if fn.name == slot_name and fn.body:
            lines = solution.splitlines()
            body = "\n".join(lines[fn.body[0].lineno - 1:])
            # DS-1000 bodies are uniformly indented under the slot's `def`, so the body keeps
            # its own valid indentation; return as-is.
            return body if body.strip() else solution
    return solution


# Postprocess strategies (final-answer extraction): OFFICIAL keeps everything before the
# first fence (xlang-ai/DS-1000 harness); CHAT also recovers the fenced answer block from
# chat/reasoning output, falling back to OFFICIAL on raw output. The potential always uses
# OFFICIAL; only the evaluator opts into CHAT.
OFFICIAL = _postprocess_official
CHAT = _postprocess_chat

# Backward-compatible alias; the default postprocess is the chat extractor.
_postprocess_code = _postprocess_chat
