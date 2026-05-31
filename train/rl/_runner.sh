#!/usr/bin/env bash
# Shared launcher for verl GRPO (Math-RL) training.
# A per-baseline runner only needs to set:
#   MODEL_PATH                 (the mid-trained / SFT-warmed checkpoint)
#   TRAIN_FILE VAL_FILES       (math train + eval split)
#   GLOBAL_BATCH N_ROLLOUT MICRO_BATCH_SIZE LR MAX_TOKENS
#   USE_CHAT_TEMPLATE          (True for instruct models, False for base models)
#   WANDB_PROJECT WANDB_NAME
# then `source` this file.
#
# Multi-node env (MASTER_NODE / NODE_LOCAL_IPS / ...) must be exported BEFORE
# sourcing common/distributed_env.sh and common/ray_bootstrap.sh.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/distributed_env.sh"

: "${MODEL_PATH:?MODEL_PATH is required (path to mid-trained / SFT-warmed ckpt)}"
: "${TRAIN_FILE:?TRAIN_FILE is required}"
: "${VAL_FILES:?VAL_FILES is required (a comma-list inside '[...]')}"
: "${GLOBAL_BATCH:=128}"
: "${N_ROLLOUT:=16}"
: "${MICRO_BATCH_SIZE:=8}"
: "${LR:=1e-6}"
: "${MAX_TOKENS:=12288}"
: "${USE_CHAT_TEMPLATE:=False}"
: "${ANSWER_REWARD:=1.0}"
: "${FORMAT_REWARD:=0.2}"
: "${SAVE_FREQ:=20}"
: "${TEST_FREQ:=5}"
: "${EPOCHS:=2}"
: "${WANDB_PROJECT:=cpt-rl}"
: "${WANDB_NAME:=$(date +%Y%m%d-%H%M%S)}"

source "$SCRIPT_DIR/../common/ray_bootstrap.sh"

NUM_WORKERS=$(( NPROC_PER_NODE * NNODES ))
MINI_BATCH_SIZE=$(( MICRO_BATCH_SIZE * NUM_WORKERS ))

# Pass per-step rewards via env into the Ray runtime
declare -A RAY_VARS=(
    [VLLM_ATTENTION_BACKEND]=XFORMERS
    [NCCL_SOCKET_IFNAME]=bond1
    [GLOO_SOCKET_IFNAME]=bond1
    [NCCL_IB_GID_INDEX]=3
    [NCCL_IB_SL]=3
    [NCCL_IB_DISABLE]=0
    [NCCL_IB_CUDA_SUPPORT]=1
    [NCCL_NET_GDR_LEVEL]=2
    [NCCL_IB_QPS_PER_CONNECTION]=4
    [NCCL_IB_TC]=160
    [NCCL_IB_TIMEOUT]=22
    [WANDB_PROJECT]=$WANDB_PROJECT
    [WANDB_MODE]=$WANDB_MODE
    [VERL_ANSWER_REWARD]=$ANSWER_REWARD
    [VERL_FORMAT_REWARD]=$FORMAT_REWARD
)
env_json=""
for k in "${!RAY_VARS[@]}"; do
    env_json="$env_json,\"$k\":\"${RAY_VARS[$k]}\""
done
RUNTIME_ENV="{\"env_vars\":{${env_json:1}}}"

# Only rank 0 submits the job; workers idle and exit when Ray stops.
if [ "$RANK" -ne 0 ]; then
    echo "[rl-runner] worker rank=$RANK idle until ray stops"
    while ray status 1>/dev/null 2>&1; do sleep 5m; done
    exit 0
fi

echo "[rl-runner] model=$MODEL_PATH"
echo "[rl-runner] train=$TRAIN_FILE"
echo "[rl-runner] bs=$GLOBAL_BATCH mini=$MINI_BATCH_SIZE micro=$MICRO_BATCH_SIZE n_roll=$N_ROLLOUT lr=$LR"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="$RUNTIME_ENV" \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$GLOBAL_BATCH \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_TOKENS \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    data.use_chat_template=$USE_CHAT_TEMPLATE \
    reward_model.reward_manager=prime \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=4096 \
    reward_model.overlong_buffer.penalty_factor=0.5 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.total_training_steps=-1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1e-3 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.drop_overlong_samples=True \
    actor_rollout_ref.actor.max_overlong_keep_ratio=0.05 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=$N_ROLLOUT \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_NAME \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$EPOCHS
