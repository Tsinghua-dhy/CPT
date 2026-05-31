#!/usr/bin/env bash
# SFT-80K (volume-matched baseline) on Qwen3-4B-Base.
# Recipe: Base -> [SFT-80K (this stage)] -> Math-RL.
#
# Data: ~80K answerable-math traces, matched to the total token budget of
# CPT + Math-SFT (paper §4.1, "SFT-80K").

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export MASTER_NODE=${MASTER_NODE:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-23457}
export NODE_LOCAL_IPS=${NODE_LOCAL_IPS:-"127.0.0.1"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-4B-Base"}
export TRAIN_FILE=${TRAIN_FILE:-"$REPO_ROOT/data/math_sft_80k/train.parquet"}
export VAL_FILE=${VAL_FILE:-"$REPO_ROOT/data/math_sft_80k/test.parquet"}
export GLOBAL_BATCH=128
export MICRO_BATCH_PER_GPU=2
export LR=2e-5
export EPOCHS=2
export MAX_LEN=8192
export USE_CHAT_TEMPLATE=False
export WANDB_PROJECT=cpt-qwen3-4b
export WANDB_NAME=sft-80k-baseline-$(date +%Y%m%d)

source "$SCRIPT_DIR/_runner.sh"
