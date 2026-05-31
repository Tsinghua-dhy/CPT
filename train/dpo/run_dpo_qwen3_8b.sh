#!/usr/bin/env bash
# DPO baseline on Qwen3-8B (paper §4.3, "DPO+RL").
# Recipe: Base -> Math-SFT -> [DPO (this stage)] -> Math-RL.
# Data: 70,352 preference pairs derived from CPT-style trace pairs.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export MASTER_NODE=${MASTER_NODE:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-23457}
export NODE_LOCAL_IPS=${NODE_LOCAL_IPS:-"127.0.0.1"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
source "$SCRIPT_DIR/../common/distributed_env.sh"

# Should point at a math-SFT warm-started checkpoint (the same warm-up as SFT+RL).
SFT_CKPT=${SFT_CKPT:-"./checkpoints/math_sft_qwen3_8b"}
TRAIN_FILE=${TRAIN_FILE:-"$REPO_ROOT/data/dpo_baseline/train.parquet"}
VAL_FILE=${VAL_FILE:-"$REPO_ROOT/data/dpo_baseline/test.parquet"}

GLOBAL_BATCH=256
MICRO_BATCH_PER_GPU=1
LR=1e-6
EPOCHS=2
MAX_LEN=8192
BETA=0.1

export WANDB_PROJECT=cpt-qwen3-8b
export WANDB_NAME=dpo-baseline-$(date +%Y%m%d)

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m verl.trainer.fsdp_dpo_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.prompt_key=prompt \
    data.chosen_key=chosen \
    data.rejected_key=rejected \
    data.train_batch_size=$GLOBAL_BATCH \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    data.max_length=$MAX_LEN \
    data.truncation=right \
    data.use_chat_template=False \
    optim.lr=$LR \
    optim.warmup_steps_ratio=0.1 \
    dpo.beta=$BETA \
    dpo.loss_type=sigmoid \
    dpo.label_smoothing=0.0 \
    dpo.reference_free=False \
    model.partial_pretrain=$SFT_CKPT \
    model.strategy=fsdp \
    model.enable_gradient_checkpointing=True \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=60 \
    trainer.test_freq=20 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_NAME \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.default_hdfs_dir=null \
    "$@"
