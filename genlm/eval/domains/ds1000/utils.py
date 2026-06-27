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
    t = _strip_reasoning(t.split("</code>")[0])
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", t, re.S)
    if blocks:
        t = blocks[-1]
    else:
        t = t.replace("```python", "")
        t = t.split("```")[0]
    t = t.split("# SOLUTION END")[0]
    t = t.split("\nEND SOLUTION")[0]
    return t.replace("<code>", "")


# Postprocess strategies (final-answer extraction): OFFICIAL keeps everything before the
# first fence (xlang-ai/DS-1000 harness); CHAT also recovers the fenced answer block from
# chat/reasoning output, falling back to OFFICIAL on raw output. The potential always uses
# OFFICIAL; only the evaluator opts into CHAT.
OFFICIAL = _postprocess_official
CHAT = _postprocess_chat

# Backward-compatible alias; the default postprocess is the chat extractor.
_postprocess_code = _postprocess_chat
