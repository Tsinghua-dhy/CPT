#!/bin/bash
# Evaluate math benchmarks (MATH-500, AIME-24/25, AMC-22/23, Minerva, OlympiadBench).
#
# Usage:
#   bash eval_math.sh <model_path_or_repo>
#
# Or edit the `models` array below and run without arguments.

set -e

models=(
    # Add one or more local paths / ModelScope repo ids here, for example:
    # Tsinghuadhy/CPT-RL-Qwen3-14B
    # ./checkpoints/your-model
)

# Allow a single positional argument as an alternative to editing the array
if [ $# -ge 1 ]; then
    models=("$@")
fi

src_files=(
    "../dataset/math500/test.jsonl"
    "../dataset/aime24/test.jsonl"
    "../dataset/aime25/test.jsonl"
    "../dataset/amc22/test.jsonl"
    "../dataset/amc23/test.jsonl"
    "../dataset/minervamath/test.jsonl"
    "../dataset/olympiad/test.jsonl"
)

GPU_ID=${GPU_ID:-"0,1,2,3,4,5,6,7"}

cd "$(dirname "$0")"
for model_path in "${models[@]}"; do
    echo "====== Evaluating: $model_path ======"
    python ./eval_math.py \
        --gpu_id "$GPU_ID" \
        --temp 0.5 \
        --src_files "${src_files[@]}" \
        --model_path "$model_path" \
        --gpu_memory_rate 0.95 \
        --max_tokens 12288
done
