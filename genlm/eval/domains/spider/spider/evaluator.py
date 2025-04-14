from pathlib import Path

from . import evaluation as E
from .evaluation import (
    build_foreign_key_map_from_json,
    build_valid_col_units,
    rebuild_sql_val,
    rebuild_sql_col,
    eval_exec_match,
)


class Evaluator:
    def __init__(self, spider_dir: Path, timeout=None):
        self.tables_path = spider_dir / "tables.json"
        self.db_path = spider_dir / "database"
        self.kmaps = build_foreign_key_map_from_json(self.tables_path)
        self.official_evaluator = E.Evaluator()  # the official Spider Evaluator
        self.timeout = timeout

    def evaluate(self, gold: str, pred: str, db_name: str, return_level: bool = False):
        """Returns: bool, Optional[str]

        On success (i.e., predicted execution result is the same as gold), returns `(True, None)`
        On failure, returns `(False, reason)` where reason is one of the two cases:
        * `invalid` if `pred` sql is not a well-formed sql statement that can be parsed by sqlite
        * `mismatch` if `pred` is a well-formed sql but the execution result is different from that of the `gold`.
        """
        db = self.db_path / db_name / (db_name + ".sqlite")
        schema = E.Schema(E.get_schema(db))

        try:
            g_sql = E.get_sql(schema, gold)
            p_sql = E.get_sql(schema, pred)
        except Exception:
            # sql is ill-formed (can't be parsed by sqlite engine)
            return False, "invalid"

        kmap = self.kmaps[db_name]

        g_valid_col_units = build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)

        p_valid_col_units = build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

        exec_match = eval_exec_match(db, pred, gold, p_sql, g_sql, timeout=self.timeout)
        reason = None if exec_match else "mismatch"

        if not return_level:
            return exec_match, reason

        difficulty_level = self.official_evaluator.eval_hardness(g_sql)
        return exec_match, reason, difficulty_level
