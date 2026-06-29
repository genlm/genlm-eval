#!/usr/bin/env python
"""Generate LM rollouts on Spider 2.0-Snow with vLLM (no constrained decoding).

A *plain sampling* harness over Qwen3 models in both thinking and non-thinking
modes. It reuses only the repo's prompt formatting (``SYSTEM_PROMPT`` +
``chat_template_messages``) and the ``Spider2Dataset`` Snow loader, then samples
completions straight from vLLM. No ``genlm.control`` potential / SMC is involved.

Model x mode matrix (6 configs): {Qwen3-1.7B, Qwen3-4B, Qwen3-8B} x {think, nothink}.
"thinking" toggles Qwen3's hybrid reasoning via the chat template's
``enable_thinking`` flag; thinking traces are long, so think configs use a larger
token budget and (by default) fewer samples than nothink.

Context: Qwen3 dense models are 40,960-token native, below Snow's ~80k-token
schemas, so YaRN rope-scaling is enabled by default (``--yarn-factor 4``) to reach
~131k. Few-shot is adaptive: each prompt starts from ``--few-shot-k`` examples and
drops them until it fits the (scaled) context.

For each config it loads the model once and sweeps every temperature. Work is
sharded over instances via ``--shard-id``/``--num-shards`` so a SLURM array can
fan ``(config x shard)`` across many GPUs (see ``scripts/euler_rollouts.sbatch``).

Output: one JSONL per ``(config, temperature, shard)`` under
``<out-dir>/<config-slug>/``, one line per instance with the gold SQL and db
carried through for later scoring with ``Spider2Evaluator``.

Example (single config, debug)::

    python scripts/generate_spider2_rollouts.py \\
        --data-dir /path/to/spider2-snow --models qwen3-1.7b-nothink \\
        --samples 100 --think-samples 20 --limit 8 --num-shards 1 --shard-id 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


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

# Qwen3 dense models' native max_position_embeddings (used as the YaRN base).
QWEN3_NATIVE_CTX = 40960


# --------------------------------------------------------------------------- #
# Prompt construction (chat template, with Qwen3 thinking toggle)              #
# --------------------------------------------------------------------------- #


def _chat_prompt(tokenizer, system_prompt, few_shot, user_message, *, thinking):
    """Chat-template prompt string. ``thinking`` toggles Qwen3 reasoning."""
    from genlm.eval.util import chat_template_messages

    return tokenizer.apply_chat_template(
        chat_template_messages(system_prompt, list(few_shot), user_message),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
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
    system_prompt: str,
    thinking: bool,
    max_shots: int,
    cap_tokens: int,
) -> FittedPrompt | None:
    """Build the longest prompt (most shots) that fits within ``cap_tokens``.

    Returns ``None`` if even the 0-shot prompt is over budget (caller skips it).
    """
    examples = instance.few_shot_examples[:max_shots]
    for k in range(len(examples), -1, -1):
        text = _chat_prompt(
            tokenizer,
            system_prompt,
            examples[:k],
            instance.user_message,
            thinking=thinking,
        )
        # The chat template already injects special tokens; don't double-count.
        n_tok = len(tokenizer(text, add_special_tokens=False).input_ids)
        if n_tok <= cap_tokens:
            return FittedPrompt(text=text, n_shots=k, prompt_tokens=n_tok)
    return None


# --------------------------------------------------------------------------- #
# Main per-config generation                                                   #
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
        help="Config slug(s) to run. Default: all six.",
    )
    p.add_argument("--samples", type=int, default=100, help="Samples/instance at t>0 (nothink).")
    p.add_argument("--think-samples", type=int, default=20, help="Samples/instance at t>0 (think).")
    p.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    p.add_argument("--few-shot-k", type=int, default=3, help="Max few-shot examples (adaptively reduced).")
    p.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens (nothink).")
    p.add_argument("--think-max-tokens", type=int, default=8192, help="Max new tokens (think).")
    # Context / YaRN (Qwen3 dense is 40960 native; Snow schemas are ~80k).
    p.add_argument("--max-model-len", type=int, default=131072, help="Engine context length.")
    p.add_argument("--yarn-factor", type=float, default=4.0, help="YaRN scaling factor; <=1 disables.")
    p.add_argument("--yarn-orig", type=int, default=QWEN3_NATIVE_CTX, help="YaRN original context.")
    # Instance sharding for SLURM job arrays.
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Debug: only the first N (post-shard) instances.")
    # vLLM engine knobs.
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
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


def run_model(args, hf_id: str, slug: str, thinking: bool, instances):
    import gc

    # vLLM's FlashInfer top-k/top-p sampler JIT-compiles a CUDA kernel at runtime,
    # which requires the CUDA toolkit (nvcc). GPU nodes commonly ship only the
    # driver, so default to vLLM's native Torch sampler. Export the var to override.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from genlm.eval.domains.spider2.spider2 import SYSTEM_PROMPT

    max_tokens = args.think_max_tokens if thinking else args.max_tokens
    n_samples = args.think_samples if thinking else args.samples

    out_dir = Path(args.out_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # YaRN context extension so the ~80k-token Snow schemas fit (Qwen3 is 40960
    # native). Override BOTH rope_scaling (so the kernel applies YaRN) and
    # max_position_embeddings (so vLLM derives the extended max and accepts a
    # larger max_model_len) -- setting only rope_scaling leaves the derived max
    # at 40960, and positions past it trigger a CUDA out-of-bounds assert.
    hf_overrides = None
    if args.yarn_factor and args.yarn_factor > 1.0:
        scaled_ctx = int(args.yarn_orig * args.yarn_factor)
        hf_overrides = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": args.yarn_factor,
                "original_max_position_embeddings": args.yarn_orig,
            },
            "max_position_embeddings": scaled_ctx,
        }

    print(
        f"[{slug}] loading {hf_id} (thinking={thinking}, max_tokens={max_tokens}, "
        f"n={n_samples}, max_model_len={args.max_model_len}, yarn={hf_overrides is not None})",
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

    # Build (and fit) prompts once; they are temperature-independent.
    prompts, kept, skipped = [], [], []
    shot_hist: dict[int, int] = {}
    for inst in instances:
        fitted = fit_prompt(
            tokenizer,
            inst,
            system_prompt=SYSTEM_PROMPT,
            thinking=thinking,
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

    prompt_texts = [fp.text for fp in prompts]

    for temp in args.temperatures:
        n = 1 if temp == 0.0 else n_samples
        shard_tag = f"shard{args.shard_id:03d}-of{args.num_shards:03d}"
        out_path = out_dir / f"{slug}__t{temp:.1f}__{shard_tag}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[{slug}] t={temp}: exists, skipping ({out_path.name})", flush=True)
            continue

        sp = SamplingParams(
            n=n,
            temperature=temp,
            top_p=1.0,  # pure temperature sampling, no nucleus truncation
            max_tokens=max_tokens,
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
                            "thinking": thinking,
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
        tmp_path.rename(out_path)  # atomic: a complete file or none
        print(
            f"[{slug}] t={temp}: wrote {len(kept)} instances x {n} samples in {dt:.0f}s -> {out_path.name}",
            flush=True,
        )

    # Free the GPU before the next config in this process (if any).
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
        f"(few-shot pool excluded: {few_shot_ids}); configs={args.models}",
        flush=True,
    )

    for slug in args.models:
        hf_id, _, thinking = MODELS_BY_SLUG[slug]
        run_model(args, hf_id, slug, thinking, instances)


if __name__ == "__main__":
    main()
