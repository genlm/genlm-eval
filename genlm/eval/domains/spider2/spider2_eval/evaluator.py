import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils import StrPath


_TOLERANCE = 1e-2


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _values_equal(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), abs_tol=_TOLERANCE, rel_tol=_TOLERANCE)
    return a == b


def _vectors_match(a: List[Any], b: List[Any], ignore_order: bool) -> bool:
    if len(a) != len(b):
        return False
    if ignore_order:
        # Sort by string repr so heterogeneous values still compare deterministically.
        a = sorted(a, key=lambda v: (v is None, str(v)))
        b = sorted(b, key=lambda v: (v is None, str(v)))
    return all(_values_equal(x, y) for x, y in zip(a, b))


def compare_pandas_table(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    condition_cols: Optional[List[int]] = None,
    ignore_order: bool = True,
) -> bool:
    """Spider 2 style result comparison.

    A predicted DataFrame is considered correct when every gold column has at
    least one predicted column that matches its values (within numeric
    tolerance and optionally ignoring row order).  When ``condition_cols`` is
    non-empty only those columns of ``gold`` are required to match.
    """
    if pred is None or gold is None:
        return False
    if condition_cols:
        try:
            gold = gold.iloc[:, condition_cols]
        except IndexError:
            return False

    gold_cols = [gold.iloc[:, i].tolist() for i in range(gold.shape[1])]
    pred_cols = [pred.iloc[:, i].tolist() for i in range(pred.shape[1])]

    if not gold_cols:
        return pred.shape[1] == 0
    if len(pred_cols) < len(gold_cols):
        return False

    for gold_col in gold_cols:
        if not any(
            _vectors_match(gold_col, pred_col, ignore_order) for pred_col in pred_cols
        ):
            return False
    return True


def execute_sqlite(sqlite_path: StrPath, query: str, timeout: Optional[float] = None) -> pd.DataFrame:
    """Execute ``query`` against the given SQLite file and return the result."""
    connect_kwargs: Dict[str, Any] = {}
    if timeout is not None:
        connect_kwargs["timeout"] = timeout
    with sqlite3.connect(str(sqlite_path), **connect_kwargs) as conn:
        return pd.read_sql_query(query, conn)


@dataclass
class Spider2EvalConfig:
    """Per-instance settings from ``spider2lite_eval.jsonl``."""

    instance_id: str
    condition_cols: List[int] = field(default_factory=list)
    ignore_order: bool = True

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "Spider2EvalConfig":
        return Spider2EvalConfig(
            instance_id=obj["instance_id"],
            condition_cols=list(obj.get("condition_cols") or []),
            ignore_order=bool(obj.get("ignore_order", True)),
        )


def load_eval_configs(path: StrPath) -> Dict[str, Spider2EvalConfig]:
    """Read the eval-config JSONL into a dict keyed by ``instance_id``."""
    configs: Dict[str, Spider2EvalConfig] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cfg = Spider2EvalConfig.from_json(json.loads(line))
            configs[cfg.instance_id] = cfg
    return configs


def load_gold_results(exec_result_dir: StrPath, instance_id: str) -> List[pd.DataFrame]:
    """Load all gold execution result CSVs that match a given instance id.

    Spider 2.0-Lite may release multiple gold variants per instance, named
    ``{instance_id}.csv``, ``{instance_id}_a.csv``, ``{instance_id}_b.csv``, etc.
    """
    exec_result_dir = Path(exec_result_dir)
    results: List[pd.DataFrame] = []
    if not exec_result_dir.exists():
        return results
    for path in sorted(exec_result_dir.iterdir()):
        if not path.is_file() or path.suffix != ".csv":
            continue
        stem = path.stem
        if stem == instance_id or stem.startswith(f"{instance_id}_"):
            try:
                results.append(pd.read_csv(path))
            except Exception:
                continue
    return results


class Evaluator:
    """Spider 2.0-Lite execution-based evaluator (SQLite backend)."""

    def __init__(
        self,
        spider2_dir: StrPath,
        sqlite_dir: Optional[StrPath] = None,
        exec_result_dir: Optional[StrPath] = None,
        eval_config_path: Optional[StrPath] = None,
        timeout: Optional[float] = None,
    ):
        spider2_dir = Path(spider2_dir)
        self.spider2_dir = spider2_dir
        self.sqlite_dir = (
            Path(sqlite_dir)
            if sqlite_dir is not None
            else spider2_dir / "resource" / "databases" / "spider2-localdb"
        )
        self.exec_result_dir = (
            Path(exec_result_dir)
            if exec_result_dir is not None
            else spider2_dir / "evaluation_suite" / "gold" / "exec_result"
        )
        self.timeout = timeout

        if eval_config_path is None:
            default_cfg = (
                spider2_dir / "evaluation_suite" / "gold" / "spider2lite_eval.jsonl"
            )
            eval_config_path = default_cfg if default_cfg.exists() else None
        self.eval_configs: Dict[str, Spider2EvalConfig] = (
            load_eval_configs(eval_config_path) if eval_config_path else {}
        )

    def _sqlite_path(self, db_name: str) -> Path:
        return self.sqlite_dir / f"{db_name}.sqlite"

    def evaluate(
        self,
        pred: str,
        gold: str,
        db_name: str,
        instance_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Returns ``(is_correct, reason, difficulty)``.

        ``reason`` is one of:

        * ``invalid`` if the predicted SQL fails to execute
        * ``mismatch`` if it executes but disagrees with every gold result
        * ``None`` when the prediction is correct
        """
        sqlite_path = self._sqlite_path(db_name)
        if not sqlite_path.exists():
            return False, f"missing sqlite database `{db_name}`", None

        try:
            pred_df = execute_sqlite(sqlite_path, pred, timeout=self.timeout)
        except Exception:
            return False, "invalid", None

        cfg = (
            self.eval_configs.get(instance_id)
            if instance_id is not None
            else None
        )
        condition_cols = cfg.condition_cols if cfg else []
        ignore_order = cfg.ignore_order if cfg else True

        gold_dfs = (
            load_gold_results(self.exec_result_dir, instance_id)
            if instance_id is not None
            else []
        )
        if not gold_dfs and gold:
            try:
                gold_dfs = [execute_sqlite(sqlite_path, gold, timeout=self.timeout)]
            except Exception:
                return False, "missing gold execution", None

        if not gold_dfs:
            return False, "missing gold execution", None

        is_correct = any(
            compare_pandas_table(
                pred_df, gold_df, condition_cols=condition_cols, ignore_order=ignore_order
            )
            for gold_df in gold_dfs
        )
        return is_correct, None if is_correct else "mismatch", None
