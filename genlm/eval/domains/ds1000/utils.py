import os
import re
import textwrap


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


def _postprocess_code(t: str) -> str:
    # Process code as in https://github.com/xlang-ai/DS-1000/blob/main/test_ds1000.py.
    #
    # Guard for reasoning-model output: such models often wrap the answer in a fenced
    # ```python ... ``` block and prepend a natural-language preamble ("Here's the
    # solution:"). The original logic kept everything BEFORE the first ``` and executed
    # the prose, raising a spurious SyntaxError (the apostrophe in "Here's" -> unterminated
    # string literal). When a fenced block is present, take its contents instead (the last
    # block = the model's committed answer); fall back to the original logic otherwise.
    t = t.split("</code>")[0]
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", t, re.S)
    if blocks:
        t = textwrap.dedent(blocks[-1])
    else:
        t = t.replace("```python", "")
        t = t.split("```")[0]
    t = t.split("\nEND SOLUTION")[0]
    return t.replace("<code>", "")
