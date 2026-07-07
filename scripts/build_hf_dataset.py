#!/usr/bin/env python
"""Package Spider 2.0-Snow rollouts into a Hugging Face dataset (two configs).

Reads the per-(config, temperature, shard) rollout JSONL files and emits:

* ``rollouts`` config -- one row per generation (column names follow Samuel's
  temperature-sweep dataset for cross-analysis):
    model, thinking, temp, instance_id, sample, text, extracted_sql, finish,
    n_tokens, n_shots, passed, score, valid (eval labels null until scored)
* ``schemas`` config -- one row per instance:
    instance_id, db, question, linked_schema, linked_tables, n_linked_tables,
    external_knowledge

Schema-linking method (BM25 top-k), few-shot k, and token budgets are constant
and belong in the dataset card, not the columns.

    python scripts/build_hf_dataset.py --rollouts-dir $SCRATCH/rollouts/snow_bm25 \
        --data-dir $SCRATCH/Spider2/spider2-snow --out /tmp/hf_build            # dry build
    python scripts/build_hf_dataset.py ... --repo <user>/spider2-snow-temp-sweep --push
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Best-effort SQL out of a generation: drop reasoning, unwrap code fences."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    return text.strip()


def split_think(text: str):
    """Split a generation into (reasoning, answer) around the </think> tag.

    Non-thinking generations have no think block, so reasoning is empty and the
    whole text is the answer.
    """
    if "</think>" in text:
        head, _, answer = text.partition("</think>")
        reasoning = head.split("<think>", 1)[-1] if "<think>" in head else head
        return reasoning, answer
    return "", text


def _load_gen_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "genmod", str(here / "generate_spider2_rollouts.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["genmod"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_rollout_rows(rollouts_dir: str, tokenizer):
    """One row per generation. ``sample`` is a per-(model,thinking,temp,instance)
    running counter, so generations split across files (base shards + a tagged
    extra run + recovery) merge into 0..N-1 without collisions."""
    rows = []
    counters: dict = {}
    for fp in sorted(glob.glob(f"{rollouts_dir}/*/*__t*__*.jsonl")):
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                gens = r["generations"]
                fins = r["finish_reasons"]
                esqls = [extract_sql(g) for g in gens]
                splits = [split_think(g) for g in gens]
                reasoning = [s[0] for s in splits]
                answer = [s[1] for s in splits]

                def _ntok(strs):
                    return [len(x) for x in tokenizer(strs, add_special_tokens=False).input_ids] if strs else []

                ntoks = _ntok(gens)
                nreason = _ntok(reasoning)
                nanswer = _ntok(answer)
                key = (r["model"], bool(r["thinking"]), float(r["temperature"]), r["spider2_instance_id"])
                base = counters.get(key, 0)
                for j, g in enumerate(gens):
                    rows.append(
                        {
                            "model": r["model"],
                            "thinking": bool(r["thinking"]),
                            "temp": float(r["temperature"]),
                            "instance_id": r["spider2_instance_id"],
                            "sample": base + j,
                            "text": g,
                            "extracted_sql": esqls[j],
                            "finish": fins[j] if j < len(fins) else None,
                            "n_tokens": ntoks[j] if j < len(ntoks) else None,
                            "n_reasoning_tokens": nreason[j] if j < len(nreason) else None,
                            "n_answer_tokens": nanswer[j] if j < len(nanswer) else None,
                            "thinking_closed": "</think>" in g,
                            "has_sql": bool(esqls[j].strip()),
                            "n_shots": r.get("n_shots"),
                            # eval label (Samuel's convention); null until scored
                            "passed": None,
                            "score": None,
                            "valid": None,
                        }
                    )
                counters[key] = base + len(gens)
    return rows


def build_schema_rows(data_dir: str, link_top_k: int, gen):
    from genlm.eval.domains.spider2.spider2_eval.dialogue import (
        load_external_knowledge,
        load_spider2_data,
    )

    snow = Path(data_dir)
    db_root = snow / "resource" / "databases"
    docs = snow / "resource" / "documents"
    data = load_spider2_data(
        snow / "spider2-snow.jsonl",
        gold_sql_dir=snow / "evaluation_suite" / "gold" / "sql",
    )
    rows = []
    for d in data:
        tables = gen._gather_tables(db_root, d.schema_name)
        if tables:
            scores = gen.BM25([f"{s} {t} {ddl}" for s, t, ddl in tables]).scores(d.utterance)
            order = sorted(range(len(tables)), key=lambda i: scores[i], reverse=True)
            chosen = [tables[i] for i in order[:link_top_k]]
        else:
            chosen = []
        ek = load_external_knowledge(docs, d.external_knowledge) if docs.exists() else None
        rows.append(
            {
                "instance_id": d.instance_id,
                "db": d.schema_name,
                "question": d.utterance,
                "linked_schema": "\n\n".join(ddl for _, _, ddl in chosen),
                "linked_tables": [t for _, t, _ in chosen],
                "n_linked_tables": len(chosen),
                "external_knowledge": ek or "",
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-dir", required=True)
    ap.add_argument("--data-dir", required=True, help="spider2-snow directory")
    ap.add_argument("--link-top-k", type=int, default=10)
    ap.add_argument("--out", default=None, help="local dir to write parquet (dry build)")
    ap.add_argument("--repo", default=None, help="HF repo id, e.g. user/spider2-snow-temp-sweep")
    ap.add_argument("--config-name", default="rollouts", help="HF config name for the rollouts (e.g. 'oracle').")
    ap.add_argument("--schemas-config-name", default="schemas", help="HF config name for the schemas.")
    ap.add_argument("--skip-schemas", action="store_true", help="Do not build/push the schemas config.")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")  # shared Qwen3 tokenizer
    gen = _load_gen_module()

    roll = build_rollout_rows(args.rollouts_dir, tok)
    sch = [] if args.skip_schemas else build_schema_rows(args.data_dir, args.link_top_k, gen)
    print(f"rollouts rows: {len(roll)} | schema rows: {len(sch)}", flush=True)
    if roll:
        print("sample rollout row:", {k: (str(v)[:80] if isinstance(v, str) else v) for k, v in roll[0].items()}, flush=True)
    if sch:
        s0 = sch[0]
        print(f"sample schema row: instance={s0['instance_id']} db={s0['db']} "
              f"n_linked_tables={s0['n_linked_tables']} tables={s0['linked_tables'][:5]}", flush=True)

    # Build the Arrow table directly with pyarrow. datasets.Dataset.from_list is
    # pathologically slow on hundreds of thousands of long-text rows (it stalled
    # indefinitely at ~324k); pa.Table.from_pylist is C-level and takes seconds.
    roll_schema = pa.schema([
        ("model", pa.string()), ("thinking", pa.bool_()), ("temp", pa.float64()),
        ("instance_id", pa.string()), ("sample", pa.int64()), ("text", pa.string()),
        ("extracted_sql", pa.string()), ("finish", pa.string()), ("n_tokens", pa.int64()),
        ("n_reasoning_tokens", pa.int64()), ("n_answer_tokens", pa.int64()),
        ("thinking_closed", pa.bool_()), ("has_sql", pa.bool_()), ("n_shots", pa.int64()),
        ("passed", pa.bool_()), ("score", pa.float64()), ("valid", pa.bool_()),
    ])
    if not args.out:
        raise SystemExit("--out is required (parquet is written first, then pushed)")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(roll, schema=roll_schema), f"{args.out}/rollouts.parquet")
    if sch:
        pq.write_table(pa.Table.from_pylist(sch), f"{args.out}/schemas.parquet")
    print(f"wrote parquet to {args.out} (rollouts={len(roll)}, schemas={len(sch)})", flush=True)

    if args.push:
        if not args.repo:
            raise SystemExit("--push requires --repo")
        from datasets import Dataset
        Dataset.from_parquet(f"{args.out}/rollouts.parquet").push_to_hub(
            args.repo, config_name=args.config_name, private=args.private)
        print(f"pushed '{args.config_name}' to https://huggingface.co/datasets/{args.repo}", flush=True)
        if sch:
            Dataset.from_parquet(f"{args.out}/schemas.parquet").push_to_hub(
                args.repo, config_name=args.schemas_config_name, private=args.private)
            print(f"pushed '{args.schemas_config_name}'", flush=True)


if __name__ == "__main__":
    main()
