#!/bin/bash
#
# Full-set evaluation script for RAG conflicts
#   data: ./conflicts.jsonl
#   eval: per-task accuracy + overall accuracy, judged by a single LLM-as-a-judge
#   usage: bash eval_rag_conflicts.sh
#
# Key: each model is loaded once; the inner loop runs all selected prompt groups in the same process. 
# Output filenames carry the promptX tag so different variants don't overwrite each other. 
#

set -e

# ---------------- Model paths (uncomment as needed) ----------------
MODEL_PATH=(
)

# ---------------- Prompt variants (A/B/C/D), any combination ----------------
# A: emphasises "is the question itself under-specified?"
# B: reverses the decision order + defaults to abstaining
# C: few-shot, with 3 judgement demos
# D: original baseline (verbose Decision Procedure) 
PROMPT_VARIANTS=(B)

# ---------------- General settings ----------------
GPU_ID_0="0,1,2,3"
GPU_ID_1="4,5,6,7"
chosen_gpu_id=1

ROLLOUT_NUM=8

DATA_FILE="./conflicts.jsonl"
OUTPUT_DIR="../outputs/rag_conflicts"

if [ "$chosen_gpu_id" -eq 0 ]; then
    GPU_ID=${GPU_ID_0}
else
    GPU_ID=${GPU_ID_1}
fi

# ---------------- cd into the script directory so .py paths are relative ----------------
cd "$(dirname "$0")"

# ---------------- runevaluation ----------------
for model_path in "${MODEL_PATH[@]}"; do
    echo "================================================================="
    echo "Evaluating model: ${model_path}"
    echo "Prompt variants : ${PROMPT_VARIANTS[*]}"
    echo "================================================================="

    # note: pass ${PROMPT_VARIANTS[@]} unquoted so each value becomes its own arg (nargs='+'). 
    python eval_rag_conflicts.py \
        --model_path "${model_path}" \
        --data_file  "${DATA_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --rollout_num ${ROLLOUT_NUM} \
        --gpu_id "${GPU_ID}" \
        --gpu_memory_rate 0.95 \
        --temp 0.5 \
        --top_p 0.9 \
        --top_k 20 \
        --max_tokens 4096 \
        --prompt_variant ${PROMPT_VARIANTS[@]}

    echo "Evaluation completed for ${model_path}"
    echo "-----------------------------------------------------------------"
done

echo "All evaluations completed!"
