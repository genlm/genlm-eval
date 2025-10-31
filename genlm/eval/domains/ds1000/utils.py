import os


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
    # Process code as in https://github.com/xlang-ai/DS-1000/blob/main/test_ds1000.py
    t = t.split("</code>")[0]
    t = t.replace("```python", "")
    t = t.split("```")[0]
    t = t.split("\nEND SOLUTION")[0]
    t = t.replace("<code>", "")
    return t.strip()
