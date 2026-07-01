#!/usr/bin/env python
"""Offline (no-warehouse) analysis of generated SQL.

Adds, per rollout row, static-analysis columns that need no database:

* ``parse_valid``      -- extracted_sql parses under sqlglot's Snowflake dialect
* ``tables_grounded``  -- every referenced table exists in the BM25-linked schema
* ``n_ref_tables`` / ``n_unknown_tables``   -- referenced vs. hallucinated tables
* ``columns_grounded`` -- referenced columns exist in the linked schema (best-effort:
                          union of columns, ignores per-table resolution / aliases)
* ``n_ref_columns`` / ``n_unknown_columns``

Groundedness is checked against the schema the model actually saw (the linked
tables from the ``schemas`` config). CTE / subquery alias names are excluded from
"tables" so they aren't counted as hallucinated.

Reads the built parquet (rollouts + schemas), adds the columns, and optionally
pushes the updated ``rollouts`` config. Requires no re-tokenization.

    python scripts/score_offline.py --build-dir $SCRATCH/hf_final \
        --repo vxef/spider2-snow-temperature-sweep --push
"""

from __future__ import annotations

import argparse
import signal

import sqlglot
from sqlglot import exp


class _ParseTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _ParseTimeout()


# Per-row parse cap: sqlglot can parse some degenerate generations exponentially
# (even under the length cap), which wedges the .map. SIGALRM aborts any single
# parse that runs long. Works in datasets' forked map workers (Linux).
signal.signal(signal.SIGALRM, _on_alarm)


def schema_map_from_ddl(ddl_text: str) -> dict:
    """Parse concatenated CREATE TABLE DDL into {table_lower: set(col_lower)}."""
    out: dict = {}
    if not ddl_text:
        return out
    try:
        stmts = sqlglot.parse(ddl_text, dialect="snowflake")
    except Exception:
        return out
    for st in stmts or []:
        if not isinstance(st, exp.Create):
            continue
        schema = st.this
        if not isinstance(schema, exp.Schema) or schema.this is None:
            continue
        name = (schema.this.name or "").lower()
        cols = {c.name.lower() for c in schema.expressions if isinstance(c, exp.ColumnDef) and c.name}
        if name:
            out[name] = cols
    return out


def analyze(sql: str, schema: dict) -> dict:
    """Static metrics for one SQL string vs a {table: {cols}} schema."""
    res = {
        "parse_valid": False,
        "tables_grounded": None,
        "n_ref_tables": None,
        "n_unknown_tables": None,
        "columns_grounded": None,
        "n_ref_columns": None,
        "n_unknown_columns": None,
    }
    if not sql or not sql.strip() or len(sql) > 20000:
        return res
    try:
        signal.setitimer(signal.ITIMER_REAL, 3.0)  # abort a pathological parse
        tree = sqlglot.parse_one(sql, dialect="snowflake")
        if tree is None:
            return res
        res["parse_valid"] = True

        prov_tables = set(schema.keys())
        prov_cols = set().union(*schema.values()) if schema else set()

        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE) if c.alias_or_name}
        ref_tables = {(t.name or "").lower() for t in tree.find_all(exp.Table) if t.name}
        ref_tables -= cte_names
        ref_tables.discard("")
        unknown_t = ref_tables - prov_tables

        ref_cols = {(c.name or "").lower() for c in tree.find_all(exp.Column) if c.name}
        ref_cols.discard("")
        ref_cols.discard("*")
        unknown_c = ref_cols - prov_cols

        res.update(
            n_ref_tables=len(ref_tables),
            n_unknown_tables=len(unknown_t),
            tables_grounded=(len(unknown_t) == 0),
            n_ref_columns=len(ref_cols),
            n_unknown_columns=len(unknown_c),
            columns_grounded=(len(unknown_c) == 0),
        )
        return res
    except Exception:
        return res
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True, help="dir with rollouts.parquet + schemas.parquet")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--num-proc", type=int, default=8)
    args = ap.parse_args()

    from datasets import Dataset

    sch = Dataset.from_parquet(f"{args.build_dir}/schemas.parquet")
    smap = {r["instance_id"]: schema_map_from_ddl(r["linked_schema"]) for r in sch}
    print(f"schemas: {len(smap)} instances", flush=True)

    roll = Dataset.from_parquet(f"{args.build_dir}/rollouts.parquet")
    print(f"rollouts: {len(roll)} rows -> analyzing", flush=True)

    def add(row):
        return analyze(row["extracted_sql"], smap.get(row["instance_id"], {}))

    roll = roll.map(add, num_proc=args.num_proc, desc="offline SQL analysis")

    # quick summary
    pv = sum(roll["parse_valid"])
    tg = sum(1 for x in roll["tables_grounded"] if x)
    print(f"parse_valid: {pv}/{len(roll)} | tables_grounded: {tg}/{len(roll)}", flush=True)

    roll.to_parquet(f"{args.build_dir}/rollouts.parquet")
    print("rewrote rollouts.parquet with offline columns", flush=True)

    if args.push:
        if not args.repo:
            raise SystemExit("--push requires --repo")
        roll.push_to_hub(args.repo, config_name="rollouts")
        print(f"pushed rollouts to {args.repo}", flush=True)


if __name__ == "__main__":
    main()
