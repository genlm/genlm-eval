#!/bin/bash
# Spider 2.0-Snow rollout generation as a SLURM job array on Euler (ETH).
#
# Each array task = one (config, instance-shard): loads one Qwen3 config and
# samples completions for its slice of instances at the configured temperature(s).
#
# Schema is reduced per-instance via BM25 question->table linking (small prompts,
# so no YaRN, fits 40,960 context). CONFIGS (6): {Qwen3-4B, Qwen3-8B, Qwen3-1.7B}
# x {think, nothink}, NUM_SHARDS=32 -> 6*32 = 192 tasks.
#
#   idx 0  qwen3-4b-think      } 4B + 1.7B fit a 24GB RTX 4090.
#   idx 1  qwen3-4b-nothink    }
#   idx 2  qwen3-8b-think      } 8B: decode KV (samples x max_tokens) needs a
#   idx 3  qwen3-8b-nothink    }     >=80GB GPU even with small prompts.
#   idx 4  qwen3-1.7b-think    }
#   idx 5  qwen3-1.7b-nothink  }
#
# Submit in three ranges (CLI flags override the #SBATCH GPU default):
#   sbatch --array=0-63    --gpus=rtx_4090:1          scripts/euler_rollouts.sbatch   # 4B
#   sbatch --array=64-127  --gpus=1 --gres=gpumem:80g scripts/euler_rollouts.sbatch   # 8B
#   sbatch --array=128-191 --gpus=rtx_4090:1          scripts/euler_rollouts.sbatch   # 1.7B

#SBATCH --job-name=spider2snow-rollouts
#SBATCH --output=logs/rollout_%A_%a.out
#SBATCH --error=logs/rollout_%A_%a.err
#SBATCH --array=0-191                 # override per submission (see above)
#SBATCH --gpus=rtx_4090:1             # default for 4B/1.7B; override with --gpus=1 --gres=gpumem:80g for 8B
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G              # 24GB/task: lean, to fit more tasks under the per-user RAM quota
#SBATCH --time=24:00:00

set -euo pipefail

# ---- user config ---------------------------------------------------------- #
NUM_SHARDS="${NUM_SHARDS:-32}"             # export NUM_SHARDS=4 for a cheap 1/instance ballpark
DATA_DIR="$SCRATCH/Spider2/spider2-snow"
OUT_DIR="${OUT_DIR:-$SCRATCH/rollouts/snow_bm25}"   # reused: completed shards are skipped
SCHEMA_SCOPE="${SCHEMA_SCOPE:-bm25}"       # export SCHEMA_SCOPE=oracle for the gold-tables skyline
LINK_TOP_K="${LINK_TOP_K:-10}"             # export LINK_TOP_K=20 for the k=20 arm
GOLD_TABLES="${GOLD_TABLES:-}"             # path to spider2-snow-gold-tables.jsonl (oracle scope)
SAMPLES="${SAMPLES:-100}"                    # nothink samples/instance at t>0
THINK_SAMPLES="${THINK_SAMPLES:-20}"         # think samples/instance at t>0
SEED="${SEED:-0}"                            # change for an independent extra batch
RUN_TAG="${RUN_TAG:-}"                       # output filename suffix for extra runs
MAX_TOKENS=1024                             # nothink generation budget
THINK_MAX_TOKENS=8192                       # think generation budget
FEW_SHOT_K=3
MAX_MODEL_LEN=40960                         # Qwen3 native; BM25 prompts fit, no YaRN
YARN_FACTOR=1.0                             # disabled
TEMPERATURES="0.6"                          # single operating point for now
TP="${TP:-1}"                               # tensor-parallel size (export TP=2 for 8B on 2x4090)

export HF_HOME="$SCRATCH/hf_cache"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

module load eth_proxy
source "$SCRATCH/venvs/spider/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0         # native sampler (no nvcc on GPU nodes)

# Per-task node-local JIT caches: shared Triton/torch caches on Lustre race under
# many concurrent tasks -> "[Errno 116] Stale file handle" at kernel compile.
TASK_CACHE="${TMPDIR:-/tmp/$USER}/jitcache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TRITON_CACHE_DIR="$TASK_CACHE/triton"
export VLLM_CACHE_ROOT="$TASK_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
# --------------------------------------------------------------------------- #

MODEL_SLUGS=(
  qwen3-4b-think         # idx 0  ] 4B  -> array 0-63,    RTX 4090
  qwen3-4b-nothink       # idx 1  ]
  qwen3-8b-think         # idx 2  ] 8B  -> array 64-127,  >=80GB
  qwen3-8b-nothink       # idx 3  ]
  qwen3-1.7b-think       # idx 4  ] 1.7B-> array 128-191, RTX 4090
  qwen3-1.7b-nothink     # idx 5  ]
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_IDX=$(( TASK_ID / NUM_SHARDS ))
SHARD_ID=$(( TASK_ID % NUM_SHARDS ))
MODEL_SLUG=${MODEL_SLUGS[$MODEL_IDX]}

mkdir -p logs "$OUT_DIR"
echo "task $TASK_ID -> config=$MODEL_SLUG shard=$SHARD_ID/$NUM_SHARDS on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python scripts/generate_spider2_rollouts.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --models "$MODEL_SLUG" \
  --schema-scope "$SCHEMA_SCOPE" \
  --link-top-k "$LINK_TOP_K" \
  --gold-tables "$GOLD_TABLES" \
  --samples "$SAMPLES" \
  --think-samples "$THINK_SAMPLES" \
  --max-tokens "$MAX_TOKENS" \
  --think-max-tokens "$THINK_MAX_TOKENS" \
  --few-shot-k "$FEW_SHOT_K" \
  --temperatures $TEMPERATURES \
  --max-model-len "$MAX_MODEL_LEN" \
  --yarn-factor "$YARN_FACTOR" \
  --num-shards "$NUM_SHARDS" \
  --shard-id "$SHARD_ID" \
  --run-tag "$RUN_TAG" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --seed "$SEED"
