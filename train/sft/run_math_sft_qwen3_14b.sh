#!/usr/bin/env bash
# Vanilla Math-SFT warm-up on Qwen3-14B-Base.
# Recipe (full SFT+RL baseline): Base -> [Math-SFT (this stage)] -> Math-RL.
#
# Data: ~9.5K math reasoning traces used as the SFT warm-up corpus (paper §4.1).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export MASTER_NODE=${MASTER_NODE:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-23457}
export NODE_LOCAL_IPS=${NODE_LOCAL_IPS:-"127.0.0.1"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-14B-Base"}
export TRAIN_FILE=${TRAIN_FILE:-"$REPO_ROOT/data/math_sft/train.parquet"}
export VAL_FILE=${VAL_FILE:-"$REPO_ROOT/data/math_sft/test.parquet"}
export GLOBAL_BATCH=64
export MICRO_BATCH_PER_GPU=1
export LR=1e-5
export EPOCHS=2
export MAX_LEN=8192
export USE_CHAT_TEMPLATE=False
export WANDB_PROJECT=cpt-qwen3-14b
export WANDB_NAME=math-sft-baseline-$(date +%Y%m%d)

source "$SCRIPT_DIR/_runner.sh"
