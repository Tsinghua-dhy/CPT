#!/usr/bin/env bash
# Shared Ray cluster bootstrap for GRPO / RL training.
# Source this AFTER common/distributed_env.sh.
# It starts a Ray head on rank 0 and joins workers on rank > 0.

: "${RAY_PORT:=6380}"

ray stop --force 2>/dev/null || true
sleep 2

check_port() {
    (echo > /dev/tcp/$MASTER_ADDR/$RAY_PORT) >/dev/null 2>&1
    return $?
}

if [ "$RANK" -eq 0 ]; then
    echo "[ray_bootstrap] starting Ray head on $MASTER_ADDR:$RAY_PORT"
    ray start --head --port $RAY_PORT
else
    echo "[ray_bootstrap] worker $RANK waiting for head at $MASTER_ADDR:$RAY_PORT"
    while ! check_port; do
        sleep 10s
    done
    while ! ray start --address=$MASTER_ADDR:$RAY_PORT; do
        echo "[ray_bootstrap] ray start failed, retrying ..."
        sleep 5s
    done
fi
echo "[ray_bootstrap] ray ready on rank $RANK"

# Wait for all nodes to join (rank 0 only)
if [ "$RANK" -eq 0 ]; then
    echo "[ray_bootstrap] waiting for all $NNODES nodes ..."
    while [ "$NNODES" -ne "$(ray status 2>/dev/null | grep -A 10000 'Active:' | grep -B 10000 'Pending:' | grep -c 'node_')" ]; do
        sleep 10s
    done
    echo "[ray_bootstrap] all $NNODES nodes joined"
fi
