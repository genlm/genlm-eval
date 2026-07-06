#!/usr/bin/env python
"""Agentic 2-step schema-retrieval rollouts on Spider 2.0-Snow with vLLM.

A minimal, bounded version of "let the model decide what schema to retrieve"
(vs. a fixed BM25 top-k):

* **Step 1 (retrieve).** The model is shown the QUESTION and the full list of
  table names in the database, and selects the tables it needs.
* **Step 2 (answer).** We inject the DDL of the *selected* tables and the model
  writes the SQL (same case-sensitive quoting instruction as the single-shot arm).

This gives the model agency over retrieval while staying a bounded 2-turn
exchange (not a fully iterative agent). If the model selects nothing parseable,
we fall back to BM25 top-k so the instance still yields a query (recorded).

Output JSONL is format-compatible with ``generate_spider2_rollouts.py`` (same
core keys) plus agentic fields: ``selected_tables``, ``n_selected``,
``retrieval_texts``, ``retrieval_fallback``. ``schema_scope`` is ``agentic2step``.

    python scripts/generate_spider2_agentic.py --data-dir $SCRATCH/Spider2/spider2-snow \
        --models qwen3-4b-think --temperatures 0.6 --samples 1 --think-samples 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Reuse helpers from the single-shot script (same scripts/ dir on sys.path[0]).
from generate_spider2_rollouts import (
    MODELS,
    MODELS_BY_SLUG,
    SYSTEM_PROMPT,
    BM25,
    _gather_tables,
    user_message_template,
)

DEFAULT_TEMPERATURES = [0.6]

SELECT_SYSTEM_PROMPT = (
    "You are a SQL analyst. You will be given a question and the list of tables "
    "available in a database (schema-qualified names). Your job in this step is "
    "only to choose which tables are needed to answer the question. "
    "Respond with the chosen table names exactly as shown in the list, one per "
    "line, and nothing else -- no SQL, no explanation. Choose all tables you may "
    "need, but do not choose tables that are irrelevant."
)


def select_user_message(menu: str, question: str, external_knowledge=None) -> str:
    extra = ""
    if external_knowledge:
        extra = "\nAdditional context to ground the question:\n" f"{external_knowledge.strip()}\n"
    return (
        "The database contains the following tables:\n"
        f"{menu}\n"
        f"{extra}"
        "Which of these tables do you need to answer the following question?\n"
        f"{question}\n"
        "List the table names you need (exactly as written above), one per line."
    )


def strip_think(text: str) -> str:
    """Return the post-</think> content (Qwen3 thinking block removed)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def build_menu(tables: List[Tuple[str, str, str]]):
    """Return (menu_text, qual_lookup, bare_lookup) for a DB's tables."""
    menu_lines, qual, bare = [], {}, {}
    for (s, t, ddl) in tables:
        q = f"{s}.{t}"
        menu_lines.append(q)
        qual[q.lower()] = (s, t, ddl)
        bare.setdefault(t.lower(), (s, t, ddl))
    return "\n".join(menu_lines), qual, bare


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def parse_selection(text: str, qual: dict, bare: dict, max_tables: int) -> List[Tuple[str, str, str]]:
    """Match model-named tables against the DB's real tables (order-preserving)."""
    chosen: List[Tuple[str, str, str]] = []
    seen = set()
    for tok in _IDENT.findall(strip_think(text)):
        low = tok.lower().strip(".")
        hit = qual.get(low)
        if hit is None and low in bare:
            hit = bare[low]
        if hit is None and "." in low:  # SCHEMA.TABLE where only the table matched
            hit = bare.get(low.rsplit(".", 1)[1])
        if hit is not None:
            key = (hit[0], hit[1])
            if key not in seen:
                seen.add(key)
                chosen.append(hit)
        if len(chosen) >= max_tables:
            break
    return chosen


# --------------------------------------------------------------------------- #
# Dataset loaders -- inlined from genlm.eval...spider2_eval.dialogue so this    #
# script has no genlm dependency (importing the eval framework is a slow cold   #
# start off Lustre; these are just file parsers).                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Spider2Datum:
    instance_id: str
    schema_name: str
    utterance: str
    query: str
    external_knowledge: Optional[str] = None


def _load_spider2_data(data_filepath: Path, gold_sql_dir: Path) -> List[Spider2Datum]:
    """Parse spider2-snow.jsonl (+ per-instance gold SQL). Accepts Lite (``db``/
    ``question``) and Snow (``db_id``/``instruction``) field names."""
    data: List[Spider2Datum] = []
    with open(data_filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gold = ""
            if gold_sql_dir is not None:
                p = Path(gold_sql_dir) / f"{obj['instance_id']}.sql"
                if p.exists():
                    gold = p.read_text(encoding="utf-8")
            data.append(
                Spider2Datum(
                    instance_id=obj["instance_id"],
                    schema_name=obj.get("db", obj.get("db_id")),
                    utterance=obj.get("question", obj.get("instruction")),
                    query=gold,
                    external_knowledge=obj.get("external_knowledge"),
                )
            )
    return data


def _load_external_knowledge(documents_dir: Path, document_name: Optional[str]) -> Optional[str]:
    if not document_name:
        return None
    p = Path(documents_dir) / document_name
    return p.read_text(encoding="utf-8") if p.exists() else None


@dataclass
class AgenticInstance:
    spider2_instance_id: str
    instance_id: int
    db: str
    gold: str
    question: str
    external_knowledge: Optional[str]
    tables: List[Tuple[str, str, str]]  # all tables in the DB


def prepare(data_dir: str, few_shot_k: int, shard_id: int, num_shards: int, limit):
    snow = Path(data_dir)
    db_root = snow / "resource" / "databases"
    docs_dir = snow / "resource" / "documents"
    data = _load_spider2_data(
        snow / "spider2-snow.jsonl",
        gold_sql_dir=snow / "evaluation_suite" / "gold" / "sql",
    )
    pool_ids = set(range(few_shot_k))  # exclude the same instances the single-shot arm reserves

    # Shard by instance index BEFORE the expensive per-instance DDL gather so each
    # task only reads its own shard's schemas -- avoids 24x redundant full-corpus
    # reads thrashing the Lustre metadata server.
    candidates = [(idx, d) for idx, d in enumerate(data) if idx not in pool_ids]
    candidates = candidates[shard_id::num_shards]
    if limit is not None:
        candidates = candidates[:limit]

    gen = []
    for idx, d in candidates:
        ek = _load_external_knowledge(docs_dir, d.external_knowledge) if docs_dir.exists() else None
        gen.append(
            AgenticInstance(
                spider2_instance_id=d.instance_id,
                instance_id=idx,
                db=d.schema_name,
                gold=d.query,
                question=d.utterance,
                external_knowledge=ek,
                tables=_gather_tables(db_root, d.schema_name),
            )
        )
    return gen


def _shard(instances, shard_id, num_shards, limit):
    out = instances[shard_id::num_shards]
    return out[:limit] if limit is not None else out


def bm25_fallback(inst: AgenticInstance, k: int) -> List[Tuple[str, str, str]]:
    if not inst.tables:
        return []
    docs = [f"{s} {t} {ddl}" for (s, t, ddl) in inst.tables]
    scores = BM25(docs).scores(inst.question)
    order = sorted(range(len(inst.tables)), key=lambda i: scores[i], reverse=True)
    return [inst.tables[i] for i in order[:k]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="rollouts/snow_agentic")
    p.add_argument("--models", nargs="+", default=[s for _, s, _ in MODELS], choices=[s for _, s, _ in MODELS])
    p.add_argument("--samples", type=int, default=1, help="Rollouts/instance at t>0 (nothink).")
    p.add_argument("--think-samples", type=int, default=1, help="Rollouts/instance at t>0 (think).")
    p.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    p.add_argument("--few-shot-k", type=int, default=3, help="Instances reserved (kept out; matches single-shot arm).")
    p.add_argument("--max-tables", type=int, default=20, help="Cap on selected tables (bounds step-2 context).")
    p.add_argument("--fallback-k", type=int, default=20, help="BM25 top-k used when the model selects nothing.")
    p.add_argument("--select-max-tokens", type=int, default=256, help="Step-1 budget (nothink).")
    p.add_argument("--select-think-max-tokens", type=int, default=3072, help="Step-1 budget (think).")
    p.add_argument("--max-tokens", type=int, default=1024, help="Step-2 SQL budget (nothink).")
    p.add_argument("--think-max-tokens", type=int, default=8192, help="Step-2 SQL budget (think).")
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--run-tag", default="")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run_model(args, hf_id, slug, thinking, gen):
    import gc

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sel_max = args.select_think_max_tokens if thinking else args.select_max_tokens
    ans_max = args.think_max_tokens if thinking else args.max_tokens
    n_samples = args.think_samples if thinking else args.samples

    out_dir = Path(args.out_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{slug}] loading {hf_id} (thinking={thinking}, sel_max={sel_max}, ans_max={ans_max}, n={n_samples})", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id)
    llm = LLM(
        model=hf_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed,
    )

    def chat(system, user, thinking):
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

    # keep instances whose step-1 (menu) prompt fits the budget
    kept, sel_prompts, menus = [], [], []
    sel_cap = args.max_model_len - sel_max - 16
    for inst in gen:
        if not inst.tables:
            continue
        menu, qual, bare = build_menu(inst.tables)
        user = select_user_message(menu, inst.question, inst.external_knowledge)
        text = chat(SELECT_SYSTEM_PROMPT, user, thinking)
        n_tok = len(tok(text, add_special_tokens=False).input_ids)
        if n_tok > sel_cap:
            continue
        kept.append(inst)
        sel_prompts.append(text)
        menus.append((qual, bare))
    print(f"[{slug}] {len(kept)} instances fit step-1 menu (of {len(gen)})", flush=True)

    for temp in args.temperatures:
        n = 1 if temp == 0.0 else n_samples
        shard_tag = f"shard{args.shard_id:03d}-of{args.num_shards:03d}"
        if args.run_tag:
            shard_tag = f"{shard_tag}__{args.run_tag}"
        out_path = out_dir / f"{slug}__t{temp:.1f}__{shard_tag}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[{slug}] t={temp}: exists, skipping ({out_path.name})", flush=True)
            continue

        # ---- Step 1: table selection (n selections per instance) ----
        t0 = time.time()
        sel_sp = SamplingParams(n=n, temperature=temp, top_p=1.0, max_tokens=sel_max, seed=args.seed)
        sel_out = llm.generate(sel_prompts, sel_sp)

        # ---- flatten to one step-2 prompt per (instance, sample); fit to context ----
        # The selected tables' DDL can exceed the window; drop lowest-priority
        # tables until the answer prompt fits (one oversized prompt would abort the
        # whole vLLM batch). Each table's DDL is prefixed with its qualified name so
        # the model can follow the three-part-naming instruction.
        ans_cap = args.max_model_len - ans_max - 16
        ans_prompts, meta = [], []  # meta[i] = (kept_idx, sample_j, selection_text, chosen, fallback)
        for ki, out in enumerate(sel_out):
            inst = kept[ki]
            qual, bare = menus[ki]
            for sj, o in enumerate(out.outputs):
                chosen = parse_selection(o.text, qual, bare, args.max_tables)
                fallback = False
                if not chosen:
                    chosen = bm25_fallback(inst, args.fallback_k)
                    fallback = True
                while True:
                    schema_str = "\n\n".join(f"-- {inst.db}.{s}.{t}\n{ddl}" for (s, t, ddl) in chosen)
                    user = user_message_template(schema_str, inst.question, inst.external_knowledge)
                    prompt = chat(SYSTEM_PROMPT, user, thinking)
                    if len(tok(prompt, add_special_tokens=False).input_ids) <= ans_cap:
                        break
                    if len(chosen) <= 1:  # single oversized table: hard char-truncate its DDL
                        s, t, ddl = chosen[0]
                        schema_str = f"-- {inst.db}.{s}.{t}\n{ddl[: ans_cap * 3]}"
                        user = user_message_template(schema_str, inst.question, inst.external_knowledge)
                        prompt = chat(SYSTEM_PROMPT, user, thinking)
                        break
                    chosen = chosen[:-1]  # drop lowest-priority selected table
                ans_prompts.append(prompt)
                meta.append((ki, sj, strip_think(o.text), chosen, fallback))

        # ---- Step 2: SQL generation (one gen per prompt) ----
        ans_sp = SamplingParams(n=1, temperature=temp, top_p=1.0, max_tokens=ans_max, seed=args.seed)
        ans_out = llm.generate(ans_prompts, ans_sp)
        dt = time.time() - t0

        # ---- regroup by instance ----
        by_inst = {ki: {"gens": [None] * n, "fin": [None] * n, "sel": [None] * n,
                        "nsel": [None] * n, "rtext": [None] * n, "fb": [None] * n}
                   for ki in range(len(kept))}
        for (ki, sj, rtext, chosen, fallback), out in zip(meta, ans_out):
            g = by_inst[ki]
            g["gens"][sj] = out.outputs[0].text.strip()
            g["fin"][sj] = out.outputs[0].finish_reason
            g["sel"][sj] = [f"{s}.{t}" for (s, t, _) in chosen]
            g["nsel"][sj] = len(chosen)
            g["rtext"][sj] = rtext
            g["fb"][sj] = fallback

        tmp_path = out_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for ki, inst in enumerate(kept):
                g = by_inst[ki]
                f.write(json.dumps({
                    "spider2_instance_id": inst.spider2_instance_id,
                    "instance_id": inst.instance_id,
                    "db": inst.db,
                    "gold": inst.gold,
                    "model": hf_id,
                    "thinking": thinking,
                    "schema_scope": "agentic2step",
                    "temperature": temp,
                    "n": n,
                    "max_tokens": ans_max,
                    "n_shots": 0,
                    "generations": g["gens"],
                    "finish_reasons": g["fin"],
                    "selected_tables": g["sel"],
                    "n_selected": g["nsel"],
                    "retrieval_texts": g["rtext"],
                    "retrieval_fallback": g["fb"],
                }) + "\n")
        tmp_path.rename(out_path)
        print(f"[{slug}] t={temp}: wrote {len(kept)} x {n} in {dt:.0f}s -> {out_path.name}", flush=True)

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    gen = prepare(args.data_dir, args.few_shot_k, args.shard_id, args.num_shards, args.limit)
    print(f"shard {args.shard_id}/{args.num_shards}: {len(gen)} instances; models={args.models}", flush=True)
    for slug in args.models:
        hf_id, _, thinking = MODELS_BY_SLUG[slug]
        run_model(args, hf_id, slug, thinking, gen)


if __name__ == "__main__":
    main()
