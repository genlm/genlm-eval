#!/usr/bin/env python
"""Generate LM rollouts on Spider 2.0-Snow with vLLM (no constrained decoding).

This is a *plain sampling* harness: it reuses only the repo's prompt formatting
(``SYSTEM_PROMPT`` + ``chat_template_messages``) and the ``Spider2Dataset`` Snow
loader, then samples completions straight from vLLM.  No ``genlm.control``
potential / SMC is involved -- these are unconstrained rollouts.

For each model it loads the weights **once** and sweeps every temperature
(temperature is just a ``SamplingParams`` change, so no reload).  Work is sharded
over instances via ``--shard-id``/``--num-shards`` so a SLURM job array can spread
``(model x shard)`` across many GPUs on Euler -- see ``scripts/euler_rollouts.sbatch``.

Few-shot is **adaptive**: each prompt starts from ``--few-shot-k`` examples and
drops them one at a time until it fits the model's context window (minus
``--max-tokens``).  This keeps all 3 shots on the 128K-context models while letting
the 8K Meta-Llama-3-8B fall back to fewer shots instead of crashing.

Output: one JSONL per ``(model, temperature, shard)`` under
``<out-dir>/<model-slug>/``, one line per instance::

    {"spider2_instance_id", "instance_id", "db", "gold", "model", "temperature",
     "n", "n_shots", "prompt_tokens", "generations": [...], "finish_reasons": [...]}

The gold SQL and ``db`` are carried through so the rollouts can be scored later
with ``Spider2Evaluator`` without re-loading the dataset.

Example (single model, single shard, local debug)::

    python scripts/generate_spider2_rollouts.py \\
        --data-dir /path/to/spider2-snow \\
        --models llama3.2-1b-instruct \\
        --samples 100 --limit 8 --num-shards 1 --shard-id 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


# (HF model id, output slug, is_instruct).  Edit ids here if your hub mirror differs.
MODELS: list[tuple[str, str, bool]] = [
    ("meta-llama/Meta-Llama-3-8B", "llama3-8b", False),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "llama3-8b-instruct", True),
    ("meta-llama/Llama-3.1-8B", "llama3.1-8b", False),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3.1-8b-instruct", True),
    ("meta-llama/Llama-3.2-1B", "llama3.2-1b", False),
    ("meta-llama/Llama-3.2-1B-Instruct", "llama3.2-1b-instruct", True),
]
MODELS_BY_SLUG = {slug: (hf_id, slug, instruct) for hf_id, slug, instruct in MODELS}

DEFAULT_TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# --------------------------------------------------------------------------- #
# Prompt construction (mirrors spider2.default_prompt_formatter, as a string)  #
# --------------------------------------------------------------------------- #


def _base_prompt(system_prompt: str, few_shot, user_message: str) -> str:
    """Raw-completion prompt for base (non-instruct) models, ``k`` shots."""
    parts = [system_prompt]
    if few_shot:
        parts.append(
            "\n\n".join(f"{inp}\nSQL query: {out}" for inp, out in few_shot)
        )
    parts.append(user_message + "\nSQL query:")
    return "\n\n".join(parts)


def _chat_prompt(tokenizer, system_prompt: str, few_shot, user_message: str) -> str:
    """Chat-template prompt string for instruct models, ``k`` shots."""
    from genlm.eval.util import chat_template_messages

    return tokenizer.apply_chat_template(
        chat_template_messages(system_prompt, list(few_shot), user_message),
        tokenize=False,
        add_generation_prompt=True,
    )


@dataclass
class FittedPrompt:
    text: str
    n_shots: int
    prompt_tokens: int


def fit_prompt(
    tokenizer,
    instance,
    *,
    is_instruct: bool,
    system_prompt: str,
    max_shots: int,
    cap_tokens: int,
) -> FittedPrompt | None:
    """Build the longest prompt (most shots) that fits within ``cap_tokens``.

    Returns ``None`` if even the 0-shot prompt is over budget (caller skips it).
    """
    examples = instance.few_shot_examples[:max_shots]
    for k in range(len(examples), -1, -1):
        shots = examples[:k]
        if is_instruct:
            text = _chat_prompt(tokenizer, system_prompt, shots, instance.user_message)
            # The chat template already injects BOS/headers; don't double-count.
            n_tok = len(tokenizer(text, add_special_tokens=False).input_ids)
        else:
            text = _base_prompt(system_prompt, shots, instance.user_message)
            n_tok = len(tokenizer(text, add_special_tokens=True).input_ids)
        if n_tok <= cap_tokens:
            return FittedPrompt(text=text, n_shots=k, prompt_tokens=n_tok)
    return None


# --------------------------------------------------------------------------- #
# Main per-model generation                                                    #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Path to the spider2-snow directory.")
    p.add_argument("--out-dir", default="rollouts/spider2-snow", help="Output root.")
    p.add_argument(
        "--models",
        nargs="+",
        default=[s for _, s, _ in MODELS],
        choices=[s for _, s, _ in MODELS],
        help="Model slug(s) to run. Default: all six.",
    )
    p.add_argument("--samples", type=int, default=100, help="Samples per instance at t>0 (t=0 is always 1, greedy).")
    p.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    p.add_argument("--few-shot-k", type=int, default=3, help="Max few-shot examples (adaptively reduced to fit context).")
    p.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens per generation.")
    # Instance sharding for SLURM job arrays.
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Debug: only the first N (post-shard) instances.")
    # vLLM engine knobs.
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None, help="Override engine context length.")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (reproducible reruns).")
    p.add_argument("--safety-margin", type=int, default=16, help="Token slack left below the context cap.")
    p.add_argument("--overwrite", action="store_true", help="Regenerate even if the output JSONL exists.")
    return p.parse_args()


def build_instances(data_dir: str, few_shot_k: int):
    """Load Snow, attaching the first ``few_shot_k`` instances as few-shot pool.

    Those few-shot instances are then excluded from the generation set to avoid
    feeding a question its own gold answer (Snow has no separate train split).
    """
    from genlm.eval.domains.spider2 import Spider2Dataset

    few_shot_ids = list(range(few_shot_k))
    dataset = Spider2Dataset.from_spider2_snow_dir(
        data_dir, few_shot_example_ids=few_shot_ids
    )
    few_shot_set = set(few_shot_ids)
    instances = [inst for inst in dataset if inst.instance_id not in few_shot_set]
    return instances, sorted(few_shot_set)


def run_model(args, hf_id: str, slug: str, is_instruct: bool, instances):
    import gc

    # vLLM's FlashInfer top-k/top-p sampler JIT-compiles a CUDA kernel at runtime,
    # which requires the CUDA toolkit (nvcc). GPU nodes commonly ship only the
    # driver, so default to vLLM's native Torch sampler. Export the var to override.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from genlm.eval.domains.spider2.spider2 import SYSTEM_PROMPT

    out_dir = Path(args.out_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{slug}] loading {hf_id} (tp={args.tensor_parallel_size}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    llm = LLM(
        model=hf_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed,
    )
    ctx = getattr(llm.llm_engine.model_config, "max_model_len", None) or args.max_model_len
    cap = ctx - args.max_tokens - args.safety_margin
    print(f"[{slug}] context={ctx}, prompt budget={cap} tokens", flush=True)

    # Build (and fit) prompts once; they are temperature-independent.
    prompts, kept, skipped = [], [], []
    shot_hist: dict[int, int] = {}
    for inst in instances:
        fitted = fit_prompt(
            tokenizer,
            inst,
            is_instruct=is_instruct,
            system_prompt=SYSTEM_PROMPT,
            max_shots=args.few_shot_k,
            cap_tokens=cap,
        )
        if fitted is None:
            skipped.append(inst.spider2_instance_id)
            continue
        prompts.append(fitted)
        kept.append(inst)
        shot_hist[fitted.n_shots] = shot_hist.get(fitted.n_shots, 0) + 1

    print(
        f"[{slug}] {len(kept)} instances fit, {len(skipped)} skipped (over context even at 0-shot). "
        f"shots used: {dict(sorted(shot_hist.items()))}",
        flush=True,
    )
    if skipped:
        print(f"[{slug}] skipped ids: {skipped}", flush=True)

    stop = ["Here is a database schema:"] if not is_instruct else None
    prompt_texts = [fp.text for fp in prompts]

    for temp in args.temperatures:
        n = 1 if temp == 0.0 else args.samples
        shard_tag = f"shard{args.shard_id:03d}-of{args.num_shards:03d}"
        out_path = out_dir / f"{slug}__t{temp:.1f}__{shard_tag}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[{slug}] t={temp}: exists, skipping ({out_path.name})", flush=True)
            continue

        sp = SamplingParams(
            n=n,
            temperature=temp,
            top_p=1.0,  # pure temperature sampling, no nucleus truncation
            max_tokens=args.max_tokens,
            stop=stop,
            seed=args.seed,
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
                            "db": inst.schema_name,
                            "gold": inst.gold,
                            "model": hf_id,
                            "temperature": temp,
                            "n": n,
                            "n_shots": fp.n_shots,
                            "prompt_tokens": fp.prompt_tokens,
                            "generations": [o.text.strip() for o in out.outputs],
                            "finish_reasons": [o.finish_reason for o in out.outputs],
                        }
                    )
                    + "\n"
                )
        tmp_path.rename(out_path)  # atomic: a complete file or none
        print(
            f"[{slug}] t={temp}: wrote {len(kept)} instances x {n} samples in {dt:.0f}s -> {out_path.name}",
            flush=True,
        )

    # Free the GPU before the next model in this process (if any).
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

    instances, few_shot_ids = build_instances(args.data_dir, args.few_shot_k)
    # Strided shard so each GPU gets an even mix of easy/hard instances.
    instances = instances[args.shard_id :: args.num_shards]
    if args.limit is not None:
        instances = instances[: args.limit]

    print(
        f"shard {args.shard_id}/{args.num_shards}: {len(instances)} instances "
        f"(few-shot pool excluded: {few_shot_ids}); models={args.models}",
        flush=True,
    )

    for slug in args.models:
        hf_id, _, is_instruct = MODELS_BY_SLUG[slug]
        run_model(args, hf_id, slug, is_instruct, instances)


if __name__ == "__main__":
    main()
