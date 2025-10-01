from pathlib import Path
import importlib.resources as ir
import shutil

_REGISTRY = {"blocksworld": "blocksworld.pddl"}
_PKG = "genlm.eval.domains.goal_inference.pddl_domains"

def get_domain_text(name: str) -> str:
    fname = _REGISTRY[name.lower()]
    with ir.files(_PKG).joinpath(fname).open("r", encoding="utf-8") as f:
        return f.read()

def materialize_domain(name: str, dst_dir: str | Path = ".cache/domains") -> Path:
    fname = _REGISTRY[name.lower()]
    dst_dir = Path(dst_dir); dst_dir.mkdir(parents=True, exist_ok=True)
    dst = (dst_dir / fname).resolve()
    src = ir.files(_PKG).joinpath(fname)
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
    return dst
