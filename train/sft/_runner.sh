#!/usr/bin/env bash
# Shared launcher for verl FSDP SFT trainer.
# A per-baseline runner only needs to set:
#   BASE_MODEL TRAIN_FILE VAL_FILE
#   GLOBAL_BATCH MICRO_BATCH_PER_GPU LR EPOCHS MAX_LEN
#   USE_CHAT_TEMPLATE     (True for instruct models, False for base models)
#   WANDB_PROJECT WANDB_NAME
# then `source` this file.
#
# Multi-node env (MASTER_NODE / NODE_LOCAL_IPS / ...) must be exported BEFORE
# sourcing common/distributed_env.sh in the caller.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/distributed_env.sh"

: "${BASE_MODEL:?BASE_MODEL is required}"
: "${TRAIN_FILE:?TRAIN_FILE is required}"
: "${VAL_FILE:?VAL_FILE is required}"
: "${GLOBAL_BATCH:=128}"
: "${MICRO_BATCH_PER_GPU:=1}"
: "${LR:=1e-5}"
: "${EPOCHS:=2}"
: "${MAX_LEN:=8192}"
: "${USE_CHAT_TEMPLATE:=False}"
: "${SAVE_FREQ:=60}"
: "${TEST_FREQ:=10}"
: "${WANDB_PROJECT:=cpt-sft}"
: "${WANDB_NAME:=$(date +%Y%m%d-%H%M%S)}"

echo "[sft-runner] base=$BASE_MODEL train=$TRAIN_FILE val=$VAL_FILE"
echo "[sft-runner] lr=$LR bs=$GLOBAL_BATCH micro=$MICRO_BATCH_PER_GPU max_len=$MAX_LEN chat=$USE_CHAT_TEMPLATE"

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=$GLOBAL_BATCH \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    data.max_length=$MAX_LEN \
    data.truncation=left \
    data.use_chat_template=$USE_CHAT_TEMPLATE \
    optim.lr=$LR \
    model.partial_pretrain=$BASE_MODEL \
    model.strategy=fsdp \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_NAME \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.checkpoint.save_contents='[hf_model]' \
    trainer.default_hdfs_dir=null \
    "$@"
