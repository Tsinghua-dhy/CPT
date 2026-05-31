#!/usr/bin/env bash
# Math-RL (GRPO) on top of a mid-trained / SFT-warmed Llama-3.2-3B-Instruct.
# Uses the chat template (the LLaMA family is instruct-tuned),
# and the instruct-formatted train / eval splits.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export MASTER_NODE=${MASTER_NODE:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-23457}
export NODE_LOCAL_IPS=${NODE_LOCAL_IPS:-"127.0.0.1"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export MODEL_PATH=${MODEL_PATH:?"set MODEL_PATH to your SFT/CPT checkpoint"}
export TRAIN_FILE=${TRAIN_FILE:-"$REPO_ROOT/data/math_rl_instruct/train.parquet"}
export VAL_FILES=${VAL_FILES:-"[$REPO_ROOT/data/math_benchmarks_instruct/aime24.parquet,$REPO_ROOT/data/math_benchmarks_instruct/aime25.parquet,$REPO_ROOT/data/math_benchmarks_instruct/amc22.parquet,$REPO_ROOT/data/math_benchmarks_instruct/amc23.parquet,$REPO_ROOT/data/math_benchmarks_instruct/math500.parquet,$REPO_ROOT/data/math_benchmarks_instruct/minervamath.parquet,$REPO_ROOT/data/math_benchmarks_instruct/olympiad.parquet]"}

export GLOBAL_BATCH=128
export N_ROLLOUT=16
export MICRO_BATCH_SIZE=8
export LR=1e-6
export MAX_TOKENS=12288
export USE_CHAT_TEMPLATE=True
export WANDB_PROJECT=${WANDB_PROJECT:-cpt-llama-rl}
export WANDB_NAME=${WANDB_NAME:-math-rl-$(date +%Y%m%d)}

source "$SCRIPT_DIR/_runner.sh"
