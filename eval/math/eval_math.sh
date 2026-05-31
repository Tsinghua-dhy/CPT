#!/bin/bash
# Evaluate math benchmarks (MATH-500, AIME-24/25, AMC-22/23, Minerva, OlympiadBench).
#
# Usage:
#   bash eval_math.sh <model_path_or_repo> [<model2> ...]
#
# Optional env:
#   GPU_ID="0,1,2,3,4,5,6,7"        which GPUs to use (default: all 8)
#   USE_CHAT_TEMPLATE=1             force using the tokenizer's chat template
#                                   (auto-detected for instruct/chat/distill models)
#   USE_LLM_JUDGE=1                 enable GPT-as-a-judge fallback for
#                                   math500 / minervamath / olympiad
#                                   (other datasets always use rule-based eval)
#
# Or edit the `models` array below and run without arguments.

set -e

models=(
    # Add one or more local paths / ModelScope repo ids here, for example:
    # Tsinghuadhy/CPT-RL-Qwen3-14B
    # ./checkpoints/your-model
)

# Allow positional arguments as an alternative to editing the array
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

# Per-file rollout count (matches paper §4.2: 8 rollouts on AIME/AMC, 2 elsewhere)
rollout_nums=(
    2   # math500
    8   # aime24
    8   # aime25
    8   # amc22
    8   # amc23
    2   # minervamath
    2   # olympiad
)

GPU_ID=${GPU_ID:-"0,1,2,3,4,5,6,7"}

extra_flags=()
if [ "${USE_CHAT_TEMPLATE:-0}" = "1" ]; then
    extra_flags+=(--use_chat_template)
fi
if [ "${USE_LLM_JUDGE:-0}" = "1" ]; then
    extra_flags+=(--use_llm_judge)
fi

cd "$(dirname "$0")"
for model_path in "${models[@]}"; do
    echo "====== Evaluating: $model_path ======"
    python ./eval_math.py \
        --gpu_id "$GPU_ID" \
        --temp 0.5 \
        --src_files "${src_files[@]}" \
        --rollout_nums "${rollout_nums[@]}" \
        --model_path "$model_path" \
        --gpu_memory_rate 0.9 \
        --max_tokens 8192 \
        "${extra_flags[@]}"
done
