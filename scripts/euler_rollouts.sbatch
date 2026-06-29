#!/bin/bash
# Spider 2.0-Snow rollout generation as a SLURM job array on Euler (ETH).
#
# Each array task = one (model, instance-shard): it loads one model on its GPU
# and sweeps all temperatures over its slice of the instances.
#
# MODELS (4): the two 8K-context Llama-3 models are intentionally excluded --
# Snow schemas alone are ~80k tokens, so they exceed an 8K context and every
# instance would be skipped. The four below are all 128K-context.
#   idx 0  llama3.1-8b            } 8B: run on RTX PRO 6000 (96GB). An 8B + an
#   idx 1  llama3.1-8b-instruct   }     ~80k-token prompt's KV (~10GB) + 16GB
#   idx 2  llama3.2-1b            }     weights won't fit a 24GB card.
#   idx 3  llama3.2-1b-instruct   } 1B: fit comfortably on a 24GB RTX 4090.
#
# With NUM_SHARDS=32 there are 4*32 = 128 tasks. Because the 8B and 1B halves
# want different GPUs, submit the array in two ranges with different GPU flags
# (CLI flags override the #SBATCH defaults below):
#
#   # 8B models (array 0-63) on RTX PRO 6000:
#   sbatch --array=0-63   --gpus=pro_6000:1 scripts/euler_rollouts.sbatch
#   # 1B models (array 64-127) on RTX 4090s:
#   sbatch --array=64-127 --gpus=rtx_4090:1 scripts/euler_rollouts.sbatch
#
# Before submitting: edit DATA_DIR/OUT_DIR/venv path, ensure $HF_TOKEN is set,
# and pre-download the 4 models once on a login node (compute nodes reach the
# Hub only via eth_proxy; pre-caching avoids 128 tasks downloading at once):
#   source $SCRATCH/venvs/spider/bin/activate; export HF_HOME=$SCRATCH/hf_cache
#   for m in Llama-3.1-8B Llama-3.1-8B-Instruct Llama-3.2-1B Llama-3.2-1B-Instruct; do
#     huggingface-cli download meta-llama/$m --quiet; done

#SBATCH --job-name=spider2snow-rollouts
#SBATCH --output=logs/rollout_%A_%a.out
#SBATCH --error=logs/rollout_%A_%a.err
#SBATCH --array=0-127                 # override per submission (see header)
#SBATCH --gpus=pro_6000:1             # default for the 8B half; override with --gpus=rtx_4090:1 for the 1B half
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- user config ---------------------------------------------------------- #
NUM_SHARDS=32
DATA_DIR="$SCRATCH/Spider2/spider2-snow"   # <-- your downloaded spider2-snow dir
OUT_DIR="$SCRATCH/rollouts/spider2-snow"
SAMPLES=100
FEW_SHOT_K=3                                # adaptively reduced to fit context
MAX_TOKENS=1024

# Keep big model/dataset caches off your home quota.
export HF_HOME="$SCRATCH/hf_cache"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated Llama weights}"

module load eth_proxy                       # outbound HTTPS for the Hub on compute nodes
# Adjust to your Euler stack if needed (e.g. `module load stack/2024-06 python/3.11.6`).
source "$SCRATCH/venvs/spider/bin/activate"

# vLLM's FlashInfer sampler JIT-needs nvcc (absent on GPU nodes); use native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
# --------------------------------------------------------------------------- #

MODEL_SLUGS=(
  llama3.1-8b            # idx 0  ] 8B  -> array 0-63,  big GPU
  llama3.1-8b-instruct   # idx 1  ]
  llama3.2-1b            # idx 2  ] 1B  -> array 64-127, RTX 4090
  llama3.2-1b-instruct   # idx 3  ]
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_IDX=$(( TASK_ID / NUM_SHARDS ))
SHARD_ID=$(( TASK_ID % NUM_SHARDS ))
MODEL_SLUG=${MODEL_SLUGS[$MODEL_IDX]}

mkdir -p logs "$OUT_DIR"
echo "task $TASK_ID -> model=$MODEL_SLUG shard=$SHARD_ID/$NUM_SHARDS on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python scripts/generate_spider2_rollouts.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --models "$MODEL_SLUG" \
  --samples "$SAMPLES" \
  --few-shot-k "$FEW_SHOT_K" \
  --max-tokens "$MAX_TOKENS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id "$SHARD_ID" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --seed 0
