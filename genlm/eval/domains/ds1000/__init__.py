from .utils import CHAT, OFFICIAL, _postprocess_code, unwrap_redeclared_function

# The evaluator and potential pull genlm.control -> torch/vLLM (a ~270s import). Load them
# lazily so importing the lightweight postprocess (OFFICIAL/CHAT/_postprocess_code) does not
# drag in the ML stack: consumers that only postprocess can import this package cheaply.
_LAZY = {
    "DS1000Dataset": ".ds1000",
    "DS1000Evaluator": ".ds1000",
    "DS1000Instance": ".ds1000",
    "default_prompt_formatter": ".ds1000",
    "DS1000RuntimeNoErrorPotential": ".runtime_no_error_potential",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)


__all__ = [
    "DS1000Instance",
    "DS1000Dataset",
    "DS1000Evaluator",
    "DS1000RuntimeNoErrorPotential",
    "default_prompt_formatter",
    "OFFICIAL",
    "CHAT",
    "_postprocess_code",
    "unwrap_redeclared_function",
]
