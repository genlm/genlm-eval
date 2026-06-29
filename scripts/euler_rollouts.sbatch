#!/bin/bash
# Spider 2.0-Snow rollout generation as a SLURM job array on Euler (ETH).
#
# Each array task = one (model, instance-shard): it loads one model on its GPU
# and sweeps all temperatures over its slice of the instances. With 6 models and
# NUM_SHARDS shards there are 6*NUM_SHARDS tasks running across as many GPUs as
# the scheduler gives you.
#
# USAGE
#   NUM_SHARDS=8 controls the array size below -> set --array=0-47 (6*8 - 1).
#   1) edit DATA_DIR, OUT_DIR, the module/venv activation, and HF_TOKEN export
#   2) sbatch --array=0-47 scripts/euler_rollouts.sbatch
#
# Sizing: 6 models x 8 shards = 48 tasks. 8B weights are ~16 GB in bf16; the
# --gres=gpumem request below asks for a 24 GB+ GPU so 8B + KV cache fits with
# tensor_parallel_size=1. Drop NUM_SHARDS for fewer/larger jobs, raise it for more.

#SBATCH --job-name=spider2snow-rollouts
#SBATCH --output=logs/rollout_%A_%a.out
#SBATCH --error=logs/rollout_%A_%a.err
#SBATCH --array=0-47                 # = 6 models * NUM_SHARDS - 1  (keep in sync!)
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g            # ask for a 24 GB+ GPU (enough for 8B, tp=1)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- user config ---------------------------------------------------------- #
NUM_SHARDS=8
DATA_DIR="$SCRATCH/spider2-snow"          # <-- your downloaded spider2-snow dir
OUT_DIR="$SCRATCH/rollouts/spider2-snow"
SAMPLES=100
FEW_SHOT_K=3
MAX_TOKENS=2048

# Keep big model/dataset caches off your home quota.
export HF_HOME="$SCRATCH/hf_cache"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated Llama weights}"

# Euler COMPUTE nodes have no direct internet; `eth_proxy` routes outbound HTTPS
# through the cluster proxy so the job can reach the Hub. Loading it is required.
# (Recommended: still pre-download all six models once on a login node so the 48
# array tasks hit the shared $HF_HOME cache instead of all downloading at once:
#   for m in Meta-Llama-3-8B Meta-Llama-3-8B-Instruct Llama-3.1-8B \
#            Llama-3.1-8B-Instruct Llama-3.2-1B Llama-3.2-1B-Instruct; do
#     huggingface-cli download meta-llama/$m --quiet; done )
module load eth_proxy
# Adjust to your Euler stack if needed (e.g. `module load stack/2024-06 python/3.11.6`).
source "$SCRATCH/venvs/spider/bin/activate"

# Use vLLM's native sampler: the FlashInfer sampler JIT-needs nvcc, absent on
# GPU nodes (the worker sets this too, but make it explicit for the job env).
export VLLM_USE_FLASHINFER_SAMPLER=0
# --------------------------------------------------------------------------- #

MODEL_SLUGS=(
  llama3-8b
  llama3-8b-instruct
  llama3.1-8b
  llama3.1-8b-instruct
  llama3.2-1b
  llama3.2-1b-instruct
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
