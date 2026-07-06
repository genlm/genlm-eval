#!/bin/bash
# Agentic 2-step schema-retrieval rollouts on Spider 2.0-Snow.
# 6 configs (1.7B/4B/8B x think/nothink). Submit in three GPU ranges:
#   NUM_SHARDS=4 RUN_TAG=agentic sbatch --array=0-7   --gpus=rtx_4090:1                   scripts/euler_agentic.sbatch  # 4B
#   NUM_SHARDS=4 RUN_TAG=agentic sbatch --array=8-15  --gpus=rtx_4090:2 --export=ALL,TP=2 scripts/euler_agentic.sbatch  # 8B
#   NUM_SHARDS=4 RUN_TAG=agentic sbatch --array=16-23 --gpus=rtx_4090:1                   scripts/euler_agentic.sbatch  # 1.7B
#
#SBATCH --job-name=spider2snow-agentic
#SBATCH --output=logs/agentic_%A_%a.out
#SBATCH --error=logs/agentic_%A_%a.err
#SBATCH --array=0-23
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- user config ---------------------------------------------------------- #
NUM_SHARDS="${NUM_SHARDS:-4}"
DATA_DIR="$SCRATCH/Spider2/spider2-snow"
OUT_DIR="${OUT_DIR:-$SCRATCH/rollouts/snow_agentic}"
SAMPLES="${SAMPLES:-1}"
THINK_SAMPLES="${THINK_SAMPLES:-1}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-agentic}"
MAX_TABLES="${MAX_TABLES:-20}"
FALLBACK_K="${FALLBACK_K:-20}"
TEMPERATURES="${TEMPERATURES:-0.6}"
MAX_MODEL_LEN=40960
TP="${TP:-1}"

export HF_HOME="$SCRATCH/hf_cache"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

module load eth_proxy
source "$SCRATCH/venvs/spider/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_ENABLE_HF_TRANSFER=1

# Per-task node-local JIT caches (avoid Lustre stale-file-handle races).
TASK_CACHE="${TMPDIR:-/tmp/$USER}/jitcache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TRITON_CACHE_DIR="$TASK_CACHE/triton"
export VLLM_CACHE_ROOT="$TASK_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

MODEL_SLUGS=(
  qwen3-4b-think qwen3-4b-nothink       # idx 0,1 -> array 0-7
  qwen3-8b-think qwen3-8b-nothink       # idx 2,3 -> array 8-15  (TP=2)
  qwen3-1.7b-think qwen3-1.7b-nothink   # idx 4,5 -> array 16-23
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_IDX=$(( TASK_ID / NUM_SHARDS ))
SHARD_ID=$(( TASK_ID % NUM_SHARDS ))
MODEL_SLUG=${MODEL_SLUGS[$MODEL_IDX]}

mkdir -p logs "$OUT_DIR"
echo "task $TASK_ID -> config=$MODEL_SLUG shard=$SHARD_ID/$NUM_SHARDS on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python scripts/generate_spider2_agentic.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --models "$MODEL_SLUG" \
  --samples "$SAMPLES" \
  --think-samples "$THINK_SAMPLES" \
  --temperatures $TEMPERATURES \
  --max-tables "$MAX_TABLES" \
  --fallback-k "$FALLBACK_K" \
  --max-model-len "$MAX_MODEL_LEN" \
  --num-shards "$NUM_SHARDS" \
  --shard-id "$SHARD_ID" \
  --run-tag "$RUN_TAG" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --seed "$SEED"
