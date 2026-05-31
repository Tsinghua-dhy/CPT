#!/usr/bin/env bash
# Math-SFT warm-up on Llama-3.2-3B-Instruct (instruct chat template enabled).
# Recipe (SFT+RL baseline for the LLaMA family): Instruct -> Math-SFT -> Math-RL.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export MASTER_NODE=${MASTER_NODE:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-23457}
export NODE_LOCAL_IPS=${NODE_LOCAL_IPS:-"127.0.0.1"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export BASE_MODEL=${BASE_MODEL:-"meta-llama/Llama-3.2-3B-Instruct"}
export TRAIN_FILE=${TRAIN_FILE:-"$REPO_ROOT/data/math_sft_instruct/train.parquet"}
export VAL_FILE=${VAL_FILE:-"$REPO_ROOT/data/math_sft_instruct/test.parquet"}
export GLOBAL_BATCH=64
export MICRO_BATCH_PER_GPU=4
export LR=3e-5
export EPOCHS=2
export MAX_LEN=8192
export USE_CHAT_TEMPLATE=True
export WANDB_PROJECT=cpt-llama-3.2-3b
export WANDB_NAME=math-sft-baseline-$(date +%Y%m%d)

source "$SCRIPT_DIR/_runner.sh"
