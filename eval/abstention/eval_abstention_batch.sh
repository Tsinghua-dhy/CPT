#!/bin/bash
# Batch evaluation on AbstentionBench (paper §4.2).
#
# For each model in `models`, run AbstentionBench with the chosen prompt style
# (normal / abstention) on the 20-dataset merged set. Datasets larger than
# `MAX_PER_DATASET` are deterministically sub-sampled using the official
# `subsampling-indices.json` shipped with AbstentionBench.
#
# Usage:
#   bash eval_abstention_batch.sh <model_path_or_repo> [<model_path_or_repo> ...]
# Or edit `models` below.

set -e

models=(
    # Add one or more local paths / ModelScope repo ids, e.g.:
    # Tsinghuadhy/CPT-RL-Qwen3-14B
)
if [ $# -ge 1 ]; then
    models=("$@")
fi

# ---- compute config ----
GPU_ID=${GPU_ID:-"0,1,2,3,4,5,6,7"}
MAX_THREADS=${MAX_THREADS:-256}      # GPT API concurrency for LLM-judge fallback

# ---- dataset config ----
DATASET_DIR=${DATASET_DIR:-"./abstention_bench"}
INDICES_FILE=${INDICES_FILE:-"$DATASET_DIR/raw_data/subsampling-indices.json"}
MAX_PER_DATASET=${MAX_PER_DATASET:-3500}    # paper default, set to 0 to disable

# ---- evaluation config ----
# DATASETS = "all" uses the 20-dataset merged split (recommended).
# Full list: alcuna, bbq, big_bench_disambiguate, big_bench_known_unknowns,
# coconot, falseqa, gpqa_abstain, gsm8k_abstain, known_unknown_questions,
# mediq, mmlu_history_abstain, mmlu_math_abstain, moral_choice, musique,
# qaqa, qasper, situated_qa, squad2, umwp, world_sense.
DATASETS=${DATASETS:-"all"}
PROMPT_STYLES=${PROMPT_STYLES:-"normal,abstention"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/abstention_bench"}
SKIP_EXISTING=${SKIP_EXISTING:-1}

cd "$(dirname "$0")"

for model_path in "${models[@]}"; do
    echo ""
    echo "====== Evaluating: $model_path ======"

    EXTRA_ARGS=()
    [ -n "$OUTPUT_DIR" ]    && EXTRA_ARGS+=("--output_dir" "$OUTPUT_DIR")
    [ "$SKIP_EXISTING" = "1" ] && EXTRA_ARGS+=("--skip_existing")

    python ./eval_abstention.py \
        --gpu_id "$GPU_ID" \
        --model_path "$model_path" \
        --dataset "$DATASETS" \
        --dataset_dir "$DATASET_DIR" \
        --prompt_style "$PROMPT_STYLES" \
        --max_per_dataset "$MAX_PER_DATASET" \
        --indices_file "$INDICES_FILE" \
        --temp 0.0 \
        --max_tokens 8192 \
        --rollout_num 1 \
        --gpu_memory_rate 0.90 \
        --max_threads "$MAX_THREADS" \
        --use_llm_judge \
        "${EXTRA_ARGS[@]}"
done
