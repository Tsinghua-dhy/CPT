#!/bin/bash

# Qwen3-4B Rollout Script
# Easy/Medium: 8x rollout, max_tokens=2048
# Hard/Very_hard: 16x rollout, max_tokens=4096
# Run 2 vLLM instances concurrently

# model list
models=(
    Qwen/Qwen3-4B-Base
    # Add more 4B model paths
)

# datafile
src_file="raw_datasets/simplerl_dapomath_final.jsonl"

# GPU config - two vLLM instances
gpu_id_0="0,1,2,3"
gpu_id_1="4,5,6,7"

# runparameter
temp=1
top_p=0.95
top_k=20
gpu_memory_rate=0.95

# Create the output directory
mkdir -p outputs/dapomath

echo "========================================"
echo "Qwen3-4B Rollout Evaluation"
echo "========================================"
echo "Models: ${#models[@]}"
echo "Data: $src_file"
echo "GPU 0: $gpu_id_0"
echo "GPU 1: $gpu_id_1"
echo "========================================"

# Run the two models concurrently
run_model() {
    local model_path=$1
    local gpu_id=$2
    local model_name=$(basename $model_path)
    
    echo ""
    echo "Starting evaluation for: $model_name"
    echo "GPU: $gpu_id"
    echo "----------------------------------------"
    
    python rollout_dapomath.py \
        --src_file "$src_file" \
        --model_path "$model_path" \
        --gpu_id "$gpu_id" \
        --temp $temp \
        --top_p $top_p \
        --top_k $top_k \
        --gpu_memory_rate $gpu_memory_rate
    
    echo "Completed: $model_name"
    echo "========================================"
}

# concurrentexecute
model_count=${#models[@]}
for ((i=0; i<model_count; i+=2)); do
    if [ $i -lt $model_count ]; then
        run_model "${models[$i]}" "$gpu_id_0" &
        pid1=$!
    fi
    
    if [ $((i+1)) -lt $model_count ]; then
        run_model "${models[$((i+1))]}" "$gpu_id_1" &
        pid2=$!
    fi
    
    # Wait for both processes to finish
    if [ ! -z "$pid1" ]; then
        wait $pid1
    fi
    if [ ! -z "$pid2" ]; then
        wait $pid2
    fi
done

echo ""
echo "All evaluations completed!"
echo "Results saved in: outputs/dapomath/"
