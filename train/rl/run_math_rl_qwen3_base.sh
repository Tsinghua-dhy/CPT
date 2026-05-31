#!/usr/bin/env bash
# Math-RL (GRPO) on top of a mid-trained / SFT-warmed Qwen3 base model.
# Paper §4.4: GRPO, group size 16, temperature 0.9, top-p 0.95,
# max response length 12288, LR 1e-6.
#
# Hook this script with MODEL_PATH = your previous SFT/CPT checkpoint.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export MASTER_NODE=${MASTER_NODE:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-23457}
export NODE_LOCAL_IPS=${NODE_LOCAL_IPS:-"127.0.0.1"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export MODEL_PATH=${MODEL_PATH:?"set MODEL_PATH to your SFT/CPT checkpoint"}
export TRAIN_FILE=${TRAIN_FILE:-"$REPO_ROOT/data/math_rl/train.parquet"}
export VAL_FILES=${VAL_FILES:-"[$REPO_ROOT/data/math_benchmarks/aime24.parquet,$REPO_ROOT/data/math_benchmarks/aime25.parquet,$REPO_ROOT/data/math_benchmarks/amc22.parquet,$REPO_ROOT/data/math_benchmarks/amc23.parquet,$REPO_ROOT/data/math_benchmarks/math500.parquet,$REPO_ROOT/data/math_benchmarks/minervamath.parquet,$REPO_ROOT/data/math_benchmarks/olympiad.parquet]"}

export GLOBAL_BATCH=128
export N_ROLLOUT=16
export MICRO_BATCH_SIZE=8
export LR=1e-6
export MAX_TOKENS=12288
export USE_CHAT_TEMPLATE=False
export WANDB_PROJECT=${WANDB_PROJECT:-cpt-qwen3-rl}
export WANDB_NAME=${WANDB_NAME:-math-rl-$(date +%Y%m%d)}

source "$SCRIPT_DIR/_runner.sh"
