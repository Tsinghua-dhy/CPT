#!/usr/bin/env bash
# CPT mid-training on OLMo-3-32B-Base (paper §3.2 and Table 3).
# Note: in the paper, OLMo-3-32B does NOT receive Math-RL afterwards; it is
# evaluated as a mid-training-only model.
# Recipe: Base -> [CPT (this stage)] -> Math-SFT (no RL).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export MASTER_NODE=${MASTER_NODE:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-23457}
export NODE_LOCAL_IPS=${NODE_LOCAL_IPS:-"127.0.0.1"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export BASE_MODEL=${BASE_MODEL:-"allenai/OLMo-2-32B"}
export TRAIN_FILE=${TRAIN_FILE:-"$REPO_ROOT/data/cpt_sft/train.parquet"}
export VAL_FILE=${VAL_FILE:-"$REPO_ROOT/data/cpt_sft/test.parquet"}
export GLOBAL_BATCH=256
export MICRO_BATCH_PER_GPU=1
export LR=2e-5
export EPOCHS=2
export MAX_LEN=12288
export USE_CHAT_TEMPLATE=False
export WANDB_PROJECT=cpt-olmo3-32b
export WANDB_NAME=cpt-mid-training-$(date +%Y%m%d)

source "$SCRIPT_DIR/_runner.sh"
