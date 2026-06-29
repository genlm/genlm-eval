import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils import StrPath, backend_for_instance


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


def execute_snowflake(
    credential: Dict[str, Any], query: str, timeout: Optional[float] = None
) -> pd.DataFrame:
    """Execute ``query`` against Snowflake and return the result.

    Spider 2.0-Lite gold/predicted SQL is fully qualified
    (``DATABASE.SCHEMA.TABLE``), so the connection only needs the participant
    ``role`` and ``warehouse`` from ``snowflake_credential.json`` -- no
    per-database ``USE`` is required.
    """
    import snowflake.connector

    connect_kwargs: Dict[str, Any] = {
        "account": credential["account"],
        "user": credential["username"],
        "password": credential["password"],
    }
    for key in ("role", "warehouse"):
        if credential.get(key):
            connect_kwargs[key] = credential[key]
    if timeout is not None:
        connect_kwargs["login_timeout"] = timeout

    conn = snowflake.connector.connect(**connect_kwargs)
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [c[0] for c in cur.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


def execute_bigquery(client, query: str, timeout: Optional[float] = None) -> pd.DataFrame:
    """Execute ``query`` against BigQuery and return the result.

    ``client`` is a ``google.cloud.bigquery.Client`` (built lazily by the
    evaluator so this module imports no Google libraries at import time).
    """
    job = client.query(query)
    result = job.result(timeout=timeout)
    return result.to_dataframe(create_bqstorage_client=False)


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
    """Spider 2.0-Lite execution-based evaluator (SQLite + Snowflake backends)."""

    def __init__(
        self,
        spider2_dir: StrPath,
        sqlite_dir: Optional[StrPath] = None,
        exec_result_dir: Optional[StrPath] = None,
        eval_config_path: Optional[StrPath] = None,
        snowflake_credential_path: Optional[StrPath] = None,
        bigquery_project: Optional[str] = None,
        bigquery_credential_path: Optional[StrPath] = None,
        timeout: Optional[float] = None,
    ):
        spider2_dir = Path(spider2_dir)
        self.spider2_dir = spider2_dir
        self.sqlite_dir = (
            Path(sqlite_dir)
            if sqlite_dir is not None
            else spider2_dir / "resource" / "databases" / "spider2-localdb"
        )

        if snowflake_credential_path is None:
            default_cred = spider2_dir / "evaluation_suite" / "snowflake_credential.json"
            snowflake_credential_path = default_cred if default_cred.exists() else None
        self.snowflake_credential_path = snowflake_credential_path
        self._snowflake_credential: Optional[Dict[str, Any]] = None

        # BigQuery is opt-in: it executes only when a project is configured.
        # With no explicit credential file it falls back to Application Default
        # Credentials (``gcloud auth application-default login``).
        self.bigquery_project = bigquery_project
        self.bigquery_credential_path = bigquery_credential_path
        self._bigquery_client = None
        self.exec_result_dir = (
            Path(exec_result_dir)
            if exec_result_dir is not None
            else spider2_dir / "evaluation_suite" / "gold" / "exec_result"
        )
        self.timeout = timeout

        if eval_config_path is None:
            # Lite ships ``spider2lite_eval.jsonl``; Snow ships
            # ``spider2snow_eval.jsonl`` in the same location. Use whichever exists.
            gold_dir = spider2_dir / "evaluation_suite" / "gold"
            for cfg_name in ("spider2lite_eval.jsonl", "spider2snow_eval.jsonl"):
                candidate = gold_dir / cfg_name
                if candidate.exists():
                    eval_config_path = candidate
                    break
        self.eval_configs: Dict[str, Spider2EvalConfig] = (
            load_eval_configs(eval_config_path) if eval_config_path else {}
        )

    def _sqlite_path(self, db_name: str) -> Path:
        return self.sqlite_dir / f"{db_name}.sqlite"

    def _get_snowflake_credential(self) -> Dict[str, Any]:
        if self._snowflake_credential is None:
            path = self.snowflake_credential_path
            if path is None or not Path(path).exists():
                raise FileNotFoundError("snowflake credential file not found")
            with open(path, encoding="utf-8") as f:
                self._snowflake_credential = json.load(f)
        return self._snowflake_credential

    def _get_bigquery_client(self):
        if self._bigquery_client is None:
            from google.cloud import bigquery

            if self.bigquery_credential_path is not None:
                self._bigquery_client = bigquery.Client.from_service_account_json(
                    str(self.bigquery_credential_path), project=self.bigquery_project
                )
            else:  # Application Default Credentials
                self._bigquery_client = bigquery.Client(project=self.bigquery_project)
        return self._bigquery_client

    def _runner(self, backend: str, db_name: str):
        """Return a ``query -> DataFrame`` callable for ``backend``.

        Returns ``(runner, None)`` on success or ``(None, reason)`` when the
        backend is unavailable (missing DB / credentials / not enabled).
        """
        if backend == "sqlite":
            sqlite_path = self._sqlite_path(db_name)
            if not sqlite_path.exists():
                return None, f"missing sqlite database `{db_name}`"
            return (lambda q: execute_sqlite(sqlite_path, q, timeout=self.timeout)), None
        if backend == "snowflake":
            try:
                credential = self._get_snowflake_credential()
            except FileNotFoundError:
                return None, "missing snowflake credentials"
            return (lambda q: execute_snowflake(credential, q, timeout=self.timeout)), None
        if backend == "bigquery":
            if self.bigquery_project is None:
                return None, "missing bigquery project"
            client = self._get_bigquery_client()
            return (lambda q: execute_bigquery(client, q, timeout=self.timeout)), None
        return None, f"backend `{backend}` not enabled"

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
        backend = backend_for_instance(instance_id) if instance_id else "sqlite"
        run, reason = self._runner(backend, db_name)
        if run is None:
            return False, reason, None

        try:
            pred_df = run(pred)
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
                gold_dfs = [run(gold)]
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
