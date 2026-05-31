#!/usr/bin/env bash
# Shared multi-node environment setup for SFT / DPO / RL training scripts.
# Source this file from a per-baseline runner; it expects MASTER_NODE,
# MASTER_PORT and NODE_LOCAL_IPS to be set (or sane defaults) by the runner.
#
# Required env (per runner):
#   MASTER_NODE      : externally visible IP of the rank-0 node
#   MASTER_PORT      : port used by torch / ray
#   NODE_LOCAL_IPS   : space-separated local IPs of all participating nodes
#                     (the index of the local host within this list = NODE_RANK)
#   NPROC_PER_NODE   : GPUs per node (default 8)
#
# Sets the following for the caller:
#   NNODES, NODE_RANK, RANK, WORLD_SIZE, MASTER_ADDR
#   plus NCCL/IB env vars for high-throughput multi-node communication.

set -e

: "${MASTER_PORT:=23457}"
: "${NPROC_PER_NODE:=8}"

if [ -z "$NODE_LOCAL_IPS" ] || [ -z "$MASTER_NODE" ]; then
    echo "[distributed_env] ERROR: set MASTER_NODE and NODE_LOCAL_IPS first."
    exit 1
fi

CURRENT_LOCAL_IP=$(hostname -I | awk '{print $1}')
NODE_IPS_ARRAY=($NODE_LOCAL_IPS)

NNODES=${#NODE_IPS_ARRAY[@]}
NODE_RANK=-1
for i in "${!NODE_IPS_ARRAY[@]}"; do
    if [ "${NODE_IPS_ARRAY[$i]}" == "$CURRENT_LOCAL_IP" ]; then
        NODE_RANK=$i
        break
    fi
done

if [ $NODE_RANK -eq -1 ]; then
    echo "[distributed_env] ERROR: $CURRENT_LOCAL_IP not in NODE_LOCAL_IPS=$NODE_LOCAL_IPS"
    exit 1
fi

# Re-export for torchrun + ray
export NNODES NODE_RANK
export RANK=$NODE_RANK
export WORLD_SIZE=$NNODES
export MASTER_ADDR=$MASTER_NODE
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

echo "[distributed_env] CURRENT=$CURRENT_LOCAL_IP RANK=$NODE_RANK / NNODES=$NNODES MASTER=$MASTER_ADDR:$MASTER_PORT"

# ----- NCCL / InfiniBand tuning (adapt the HCA names to your cluster) -----
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
# export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,...   # set this if your cluster uses RDMA

export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}

# ----- Weights & Biases (optional; set WANDB_API_KEY in your env if needed) -----
export WANDB_PROJECT=${WANDB_PROJECT:-cpt}
export WANDB_MODE=${WANDB_MODE:-online}
