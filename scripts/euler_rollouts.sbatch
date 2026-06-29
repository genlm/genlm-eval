#!/bin/bash
# Spider 2.0-Snow rollout generation as a SLURM job array on Euler (ETH).
#
# Each array task = one (config, instance-shard): it loads one Qwen3 config on
# its GPU and sweeps all temperatures over its slice of the instances.
#
# CONFIGS (6): {Qwen3-4B, Qwen3-8B, Qwen3-1.7B} x {think, nothink}. "think"
# toggles Qwen3 reasoning (chat-template enable_thinking); think uses a larger
# token budget (8192) and fewer samples (20) than nothink (1024 tokens, 100).
# YaRN extends the 40,960 native context to ~131k so ~80k-token Snow schemas fit.
#
#   idx 0  qwen3-4b-think      } 4B + 8B: run on RTX PRO 6000 (96GB). 80k-token
#   idx 1  qwen3-4b-nothink    }          prompts + long thinking traces need
#   idx 2  qwen3-8b-think      }          far more KV than a 24GB card holds.
#   idx 3  qwen3-8b-nothink    }
#   idx 4  qwen3-1.7b-think    } 1.7B: fits a 24GB RTX 4090.
#   idx 5  qwen3-1.7b-nothink  }
#
# With NUM_SHARDS=32 there are 6*32 = 192 tasks. Submit in two ranges so each
# half lands on the right GPU (CLI flags override the #SBATCH defaults):
#
#   # 4B + 8B (array 0-127) on RTX PRO 6000:
#   sbatch --array=0-127   --gpus=pro_6000:1 scripts/euler_rollouts.sbatch
#   # 1.7B (array 128-191) on RTX 4090:
#   sbatch --array=128-191 --gpus=rtx_4090:1 scripts/euler_rollouts.sbatch
#
# Before submitting: edit DATA_DIR/OUT_DIR/venv path, ensure $HF_TOKEN is set,
# and pre-download the 3 models once on a login node (compute nodes reach the
# Hub only via eth_proxy; pre-caching avoids 192 tasks downloading at once):
#   source $SCRATCH/venvs/spider/bin/activate; export HF_HOME=$SCRATCH/hf_cache
#   for m in Qwen3-1.7B Qwen3-4B Qwen3-8B; do hf download Qwen/$m --quiet; done

#SBATCH --job-name=spider2snow-rollouts
#SBATCH --output=logs/rollout_%A_%a.out
#SBATCH --error=logs/rollout_%A_%a.err
#SBATCH --array=0-191                 # override per submission (see header)
#SBATCH --gpus=pro_6000:1             # default for the 4B/8B half; override with --gpus=rtx_4090:1 for 1.7B
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- user config ---------------------------------------------------------- #
NUM_SHARDS=32
DATA_DIR="$SCRATCH/Spider2/spider2-snow"   # <-- your downloaded spider2-snow dir
OUT_DIR="$SCRATCH/rollouts/spider2-snow"
SAMPLES=100                                 # nothink samples/instance at t>0
THINK_SAMPLES=20                            # think samples/instance at t>0
MAX_TOKENS=1024                             # nothink generation budget
THINK_MAX_TOKENS=8192                       # think generation budget
FEW_SHOT_K=3                                # adaptively reduced to fit context
MAX_MODEL_LEN=131072
YARN_FACTOR=4.0                             # extend Qwen3 40960 -> ~131k

# Keep big model/dataset caches off your home quota.
export HF_HOME="$SCRATCH/hf_cache"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated weights}"

module load eth_proxy                       # outbound HTTPS for the Hub on compute nodes
# Adjust to your Euler stack if needed (e.g. `module load stack/2024-06 python/3.11.6`).
source "$SCRATCH/venvs/spider/bin/activate"

# vLLM's FlashInfer sampler JIT-needs nvcc (absent on GPU nodes); use native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
# --------------------------------------------------------------------------- #

MODEL_SLUGS=(
  qwen3-4b-think         # idx 0  ] 4B+8B -> array 0-127,  RTX PRO 6000
  qwen3-4b-nothink       # idx 1  ]
  qwen3-8b-think         # idx 2  ]
  qwen3-8b-nothink       # idx 3  ]
  qwen3-1.7b-think       # idx 4  ] 1.7B  -> array 128-191, RTX 4090
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
  --samples "$SAMPLES" \
  --think-samples "$THINK_SAMPLES" \
  --max-tokens "$MAX_TOKENS" \
  --think-max-tokens "$THINK_MAX_TOKENS" \
  --few-shot-k "$FEW_SHOT_K" \
  --max-model-len "$MAX_MODEL_LEN" \
  --yarn-factor "$YARN_FACTOR" \
  --num-shards "$NUM_SHARDS" \
  --shard-id "$SHARD_ID" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --seed 0
