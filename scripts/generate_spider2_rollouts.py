#!/usr/bin/env python
"""Generate LM rollouts on Spider 2.0-Snow with vLLM (no constrained decoding).

A *plain sampling* harness over Qwen3 models in both thinking and non-thinking
modes. It reuses the ``Spider2Dataset`` Snow loader only to parse the raw fields
(question, gold SQL, db_id, external knowledge), then builds its own prompt and
samples completions straight from vLLM. No ``genlm.control`` potential is used.

Schema in the prompt (``--schema-scope``):
* ``schema`` (default): only the schema(s) the gold SQL references -- a Snow db
  may hold several schemas (e.g. AUSTIN has 5); we include just the relevant ones.
* ``table``: only the tables referenced in the gold SQL (smallest prompts).
* ``full``: every schema in the database (~80-100k tokens; needs YaRN + big GPUs).
This mirrors how Spider 2.0 baselines avoid dumping a 1000-column schema; the
agent in the paper explores instead. ``schema``/``table`` use the gold SQL to pick
the relevant subset (an oracle schema-linking condition).

Model x mode matrix (6 configs): {Qwen3-1.7B, Qwen3-4B, Qwen3-8B} x {think, nothink}.
"thinking" toggles Qwen3 reasoning via the chat template's ``enable_thinking``;
think configs use a larger token budget and fewer samples than nothink.

Context: Qwen3 dense models are 40,960-token native; with ``--yarn-factor`` set
(>1) YaRN extends context so the largest prompts still fit. Few-shot is adaptive:
each prompt drops examples until it fits the (scaled) context.

Use ``--dry-run`` to print the prompt-token distribution (per config) without a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# (HF model id, output slug, thinking).  All are Qwen3 chat models.
MODELS: list[tuple[str, str, bool]] = [
    ("Qwen/Qwen3-4B", "qwen3-4b-think", True),
    ("Qwen/Qwen3-4B", "qwen3-4b-nothink", False),
    ("Qwen/Qwen3-8B", "qwen3-8b-think", True),
    ("Qwen/Qwen3-8B", "qwen3-8b-nothink", False),
    ("Qwen/Qwen3-1.7B", "qwen3-1.7b-think", True),
    ("Qwen/Qwen3-1.7B", "qwen3-1.7b-nothink", False),
]
MODELS_BY_SLUG = {slug: (hf_id, slug, think) for hf_id, slug, think in MODELS}

DEFAULT_TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
QWEN3_NATIVE_CTX = 40960

SYSTEM_PROMPT = (
    "You are a coding assistant helping an analyst answer questions over business data in SQL. "
    "More specifically, the analyst provides you a database schema "
    "(tables in the database along with their column names and types) "
    "and asks a complex question about the data that can be solved by issuing a SQL query. "
    "In response, you write the SQL statement that answers the question. "
    "You do not provide any commentary or explanation of what the code does, "
    "just the SQL statement ending in a semicolon."
)


# --------------------------------------------------------------------------- #
# Schema selection (relevant subset from the gold SQL)                         #
# --------------------------------------------------------------------------- #


def _read_ddl_rows(ddl_csv: Path) -> List[Tuple[str, str]]:
    """Return ``(table_name, DDL)`` rows from a Spider 2 ``DDL.csv``."""
    rows: List[Tuple[str, str]] = []
    if not ddl_csv.exists():
        return rows
    with open(ddl_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("table_name") or "").strip()
            ddl = (row.get("DDL") or "").strip()
            if name and ddl:
                rows.append((name, ddl))
    return rows


def _gather_tables(db_root: Path, db_id: str) -> List[Tuple[str, str, str]]:
    """All ``(schema_name, table_name, ddl)`` for a database, across its schemas."""
    db_dir = db_root / db_id
    out: List[Tuple[str, str, str]] = []
    if not db_dir.is_dir():
        return out
    for d in sorted(db_dir.iterdir()):
        if d.is_dir():
            for table, ddl in _read_ddl_rows(d / "DDL.csv"):
                out.append((d.name, table, ddl))
    return out


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens, splitting camelCase and snake_case identifiers."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Minimal Okapi BM25 over a small document set (no external dependency)."""

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        df: dict = {}
        for d in self.docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> List[float]:
        from collections import Counter

        q = _tokenize(query)
        out = []
        for d in self.docs:
            tf = Counter(d)
            dl = len(d)
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                    )
            out.append(s)
        return out


def build_schema_str(
    db_root: Path,
    db_id: str,
    question: str,
    gold_sql: str,
    scope: str,
    top_k: int,
) -> str:
    """Build the schema DDL text for one instance under ``scope``.

    * ``full`` -- every table in the database.
    * ``schema`` -- tables in schemas whose name appears in the gold SQL (oracle).
    * ``table`` -- only the gold-referenced tables (oracle).
    * ``bm25`` -- top-``top_k`` tables retrieved from the *question* by BM25 over
      each table's (schema, name, DDL) text. Realistic, no gold leak.
    Falls back to the broader set if nothing matches.
    """
    tables = _gather_tables(db_root, db_id)
    if not tables:
        return ""
    gold_low = (gold_sql or "").lower()

    if scope == "full":
        chosen = tables
    elif scope == "schema":
        chosen = [r for r in tables if r[0].lower() in gold_low] or tables
    elif scope == "table":
        chosen = [r for r in tables if r[1].lower() in gold_low] or tables
    elif scope == "bm25":
        docs = [f"{s} {t} {ddl}" for (s, t, ddl) in tables]
        scores = BM25(docs).scores(question)
        order = sorted(range(len(tables)), key=lambda i: scores[i], reverse=True)
        chosen = [tables[i] for i in order[:top_k]]
    else:
        raise ValueError(f"unknown schema scope: {scope}")

    return "\n\n".join(ddl for (_, _, ddl) in chosen)


def user_message_template(schema_str: str, question: str, external_knowledge=None) -> str:
    extra = ""
    if external_knowledge:
        extra = (
            "\nAdditional context to ground the question:\n"
            f"{external_knowledge.strip()}\n"
        )
    return (
        "Here is a database schema:\n"
        f"{schema_str}\n"
        f"{extra}"
        "Please write me a SQL statement that answers the following question:\n"
        f"{question}\n"
        "Remember, DO NOT provide any commentary or explanation of what the code does, "
        "just the SQL statement ending in a semicolon."
    )


# --------------------------------------------------------------------------- #
# Prompt construction (chat template, with Qwen3 thinking toggle)              #
# --------------------------------------------------------------------------- #


def _chat_prompt(tokenizer, system_prompt, few_shot, user_message, *, thinking):
    """Chat-template prompt string. ``thinking`` toggles Qwen3 reasoning."""
    messages = [{"role": "system", "content": system_prompt}]
    for example, response in few_shot:
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


@dataclass
class FittedPrompt:
    text: str
    n_shots: int
    prompt_tokens: int


def fit_prompt(
    tokenizer,
    user_message: str,
    few_shot,
    *,
    system_prompt: str,
    thinking: bool,
    max_shots: int,
    cap_tokens: int,
) -> Optional[FittedPrompt]:
    """Longest prompt (most shots) that fits ``cap_tokens``; None if 0-shot is over."""
    upper = min(max_shots, len(few_shot))
    for k in range(upper, -1, -1):
        text = _chat_prompt(
            tokenizer, system_prompt, few_shot[:k], user_message, thinking=thinking
        )
        n_tok = len(tokenizer(text, add_special_tokens=False).input_ids)
        if n_tok <= cap_tokens:
            return FittedPrompt(text=text, n_shots=k, prompt_tokens=n_tok)
    return None


# --------------------------------------------------------------------------- #
# Data prep                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class RolloutInstance:
    spider2_instance_id: str
    instance_id: int
    db: str
    gold: str
    user_message: str


def prepare(data_dir: str, few_shot_k: int, schema_scope: str, top_k: int):
    """Load Snow raw fields and build per-instance prompts.

    Reads ``spider2-snow.jsonl`` + gold SQL + documents directly (not via
    ``Spider2Dataset``, whose iterator serializes the full schema per instance --
    slow and unused here). Returns ``(gen_instances, few_shot, pool_ids)``; the
    first ``few_shot_k`` instances form the few-shot pool and are excluded from
    generation (so a question never sees its own gold answer).
    """
    from genlm.eval.domains.spider2.spider2_eval.dialogue import (
        load_external_knowledge,
        load_spider2_data,
    )

    snow = Path(data_dir)
    db_root = snow / "resource" / "databases"
    docs_dir = snow / "resource" / "documents"
    data = load_spider2_data(
        snow / "spider2-snow.jsonl",
        gold_sql_dir=snow / "evaluation_suite" / "gold" / "sql",
    )
    pool_ids = set(range(few_shot_k))

    def um(d) -> str:
        ek = (
            load_external_knowledge(docs_dir, d.external_knowledge)
            if docs_dir.exists()
            else None
        )
        schema = build_schema_str(db_root, d.schema_name, d.utterance, d.query, schema_scope, top_k)
        return user_message_template(schema, d.utterance, ek)

    few_shot = [(um(d), d.query) for idx, d in enumerate(data) if idx in pool_ids]
    gen = [
        RolloutInstance(
            spider2_instance_id=d.instance_id,
            instance_id=idx,
            db=d.schema_name,
            gold=d.query,
            user_message=um(d),
        )
        for idx, d in enumerate(data)
        if idx not in pool_ids
    ]
    return gen, few_shot, sorted(pool_ids)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", required=True, help="Path to the spider2-snow directory.")
    p.add_argument("--out-dir", default="rollouts/spider2-snow", help="Output root.")
    p.add_argument(
        "--models",
        nargs="+",
        default=[s for _, s, _ in MODELS],
        choices=[s for _, s, _ in MODELS],
    )
    p.add_argument("--schema-scope", choices=["full", "schema", "table", "bm25"], default="bm25")
    p.add_argument("--link-top-k", type=int, default=10, help="Tables to retrieve for --schema-scope bm25.")
    p.add_argument("--samples", type=int, default=100, help="Samples/instance at t>0 (nothink).")
    p.add_argument("--think-samples", type=int, default=20, help="Samples/instance at t>0 (think).")
    p.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    p.add_argument("--few-shot-k", type=int, default=3, help="Max few-shot examples (adaptively reduced).")
    p.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens (nothink).")
    p.add_argument("--think-max-tokens", type=int, default=8192, help="Max new tokens (think).")
    p.add_argument("--max-model-len", type=int, default=40960, help="Engine context length.")
    p.add_argument("--yarn-factor", type=float, default=1.0, help="YaRN scaling factor; <=1 disables.")
    p.add_argument("--yarn-orig", type=int, default=QWEN3_NATIVE_CTX)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Debug: first N (post-shard) instances.")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--safety-margin", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Report prompt-token stats (no GPU) and exit.")
    return p.parse_args()


def _shard(instances, shard_id, num_shards, limit):
    out = instances[shard_id::num_shards]
    return out[:limit] if limit is not None else out


def dry_run(args, gen, few_shot):
    """Print the prompt-token distribution per config without loading vLLM."""
    from transformers import AutoTokenizer

    for slug in args.models:
        hf_id, _, thinking = MODELS_BY_SLUG[slug]
        tok = AutoTokenizer.from_pretrained(hf_id)
        budget = args.max_model_len - (args.think_max_tokens if thinking else args.max_tokens) - args.safety_margin
        toks, shots, skipped = [], {}, 0
        for inst in gen:
            fp = fit_prompt(
                tok, inst.user_message, few_shot,
                system_prompt=SYSTEM_PROMPT, thinking=thinking,
                max_shots=args.few_shot_k, cap_tokens=budget,
            )
            if fp is None:
                skipped += 1
                continue
            toks.append(fp.prompt_tokens)
            shots[fp.n_shots] = shots.get(fp.n_shots, 0) + 1
        toks_sorted = sorted(toks)

        def pct(p):
            return toks_sorted[min(len(toks_sorted) - 1, int(p * len(toks_sorted)))] if toks_sorted else 0

        print(
            f"[{slug}] budget={budget} | n={len(toks)} skipped={skipped} | "
            f"prompt_tokens min={toks_sorted[0] if toks_sorted else 0} "
            f"median={int(statistics.median(toks)) if toks else 0} "
            f"p90={pct(0.9)} max={toks_sorted[-1] if toks_sorted else 0} | "
            f"shots={dict(sorted(shots.items()))} | "
            f">40k={sum(t > 40960 for t in toks)} >131k={sum(t > 131072 for t in toks)}",
            flush=True,
        )


def run_model(args, hf_id, slug, thinking, gen, few_shot):
    import gc

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    max_tokens = args.think_max_tokens if thinking else args.max_tokens
    n_samples = args.think_samples if thinking else args.samples

    out_dir = Path(args.out_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_overrides = None
    if args.yarn_factor and args.yarn_factor > 1.0:
        hf_overrides = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": args.yarn_factor,
                "original_max_position_embeddings": args.yarn_orig,
            },
            "max_position_embeddings": int(args.yarn_orig * args.yarn_factor),
        }

    print(
        f"[{slug}] loading {hf_id} (thinking={thinking}, max_tokens={max_tokens}, n={n_samples}, "
        f"max_model_len={args.max_model_len}, yarn={hf_overrides is not None})",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    llm = LLM(
        model=hf_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed,
        hf_overrides=hf_overrides,
    )
    ctx = getattr(llm.llm_engine.model_config, "max_model_len", None) or args.max_model_len
    cap = ctx - max_tokens - args.safety_margin
    print(f"[{slug}] context={ctx}, prompt budget={cap} tokens", flush=True)

    prompts, kept, skipped, shot_hist = [], [], [], {}
    for inst in gen:
        fp = fit_prompt(
            tokenizer, inst.user_message, few_shot,
            system_prompt=SYSTEM_PROMPT, thinking=thinking,
            max_shots=args.few_shot_k, cap_tokens=cap,
        )
        if fp is None:
            skipped.append(inst.spider2_instance_id)
            continue
        prompts.append(fp)
        kept.append(inst)
        shot_hist[fp.n_shots] = shot_hist.get(fp.n_shots, 0) + 1

    print(
        f"[{slug}] {len(kept)} instances fit, {len(skipped)} skipped. "
        f"shots used: {dict(sorted(shot_hist.items()))}",
        flush=True,
    )
    if skipped:
        print(f"[{slug}] skipped ids: {skipped}", flush=True)

    prompt_texts = [fp.text for fp in prompts]

    for temp in args.temperatures:
        n = 1 if temp == 0.0 else n_samples
        shard_tag = f"shard{args.shard_id:03d}-of{args.num_shards:03d}"
        out_path = out_dir / f"{slug}__t{temp:.1f}__{shard_tag}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[{slug}] t={temp}: exists, skipping ({out_path.name})", flush=True)
            continue

        sp = SamplingParams(
            n=n, temperature=temp, top_p=1.0, max_tokens=max_tokens, seed=args.seed
        )
        t0 = time.time()
        outputs = llm.generate(prompt_texts, sp)
        dt = time.time() - t0

        tmp_path = out_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for inst, fp, out in zip(kept, prompts, outputs):
                f.write(
                    json.dumps(
                        {
                            "spider2_instance_id": inst.spider2_instance_id,
                            "instance_id": inst.instance_id,
                            "db": inst.db,
                            "gold": inst.gold,
                            "model": hf_id,
                            "thinking": thinking,
                            "schema_scope": args.schema_scope,
                            "temperature": temp,
                            "n": n,
                            "max_tokens": max_tokens,
                            "n_shots": fp.n_shots,
                            "prompt_tokens": fp.prompt_tokens,
                            "generations": [o.text.strip() for o in out.outputs],
                            "finish_reasons": [o.finish_reason for o in out.outputs],
                        }
                    )
                    + "\n"
                )
        tmp_path.rename(out_path)
        print(
            f"[{slug}] t={temp}: wrote {len(kept)} x {n} in {dt:.0f}s -> {out_path.name}",
            flush=True,
        )

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

    gen, few_shot, pool_ids = prepare(
        args.data_dir, args.few_shot_k, args.schema_scope, args.link_top_k
    )
    gen = _shard(gen, args.shard_id, args.num_shards, args.limit)
    print(
        f"shard {args.shard_id}/{args.num_shards}: {len(gen)} instances "
        f"(few-shot pool excluded: {pool_ids}); scope={args.schema_scope}; configs={args.models}",
        flush=True,
    )

    if args.dry_run:
        dry_run(args, gen, few_shot)
        return

    for slug in args.models:
        hf_id, _, thinking = MODELS_BY_SLUG[slug]
        run_model(args, hf_id, slug, thinking, gen, few_shot)


if __name__ == "__main__":
    main()
