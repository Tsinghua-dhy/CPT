#!/usr/bin/env bash
# CPT mid-training on Qwen3-14B-Base (paper §3.2, main method).
# Recipe: Base -> [CPT (this stage)] -> Math-SFT -> Math-RL.
#
# Data: CPT-SFT-70K (70,352 pairwise reasoning-comparison samples).
# Hardware in paper: 4 nodes x 8 GPUs.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- multi-node setup (override via env when launching) ----
export MASTER_NODE=${MASTER_NODE:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-23457}
export NODE_LOCAL_IPS=${NODE_LOCAL_IPS:-"127.0.0.1"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# ---- training config ----
export BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3-14B-Base"}
export TRAIN_FILE=${TRAIN_FILE:-"$REPO_ROOT/data/cpt_sft/train.parquet"}
export VAL_FILE=${VAL_FILE:-"$REPO_ROOT/data/cpt_sft/test.parquet"}
export GLOBAL_BATCH=256
export MICRO_BATCH_PER_GPU=1
export LR=2e-5
export EPOCHS=2
export MAX_LEN=12288
export USE_CHAT_TEMPLATE=False
export WANDB_PROJECT=cpt-qwen3-14b
export WANDB_NAME=cpt-mid-training-$(date +%Y%m%d)

source "$SCRIPT_DIR/_runner.sh"
