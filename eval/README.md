# Evaluation

Scripts to reproduce the three evaluation suites used in the CPT paper.

## Layout

```
eval/
├── math/
│   ├── eval_math.py      # main script
│   └── eval_math.sh      # batch driver
├── abstention/
│   ├── eval_abstention.py
│   ├── eval_abstention_batch.sh
│   ├── abstention_detector.py
│   └── correctness_evaluator.py
├── rag/
│   ├── eval_rag_conflicts.py     # DRAGged-into-Conflicts (zero-shot RAG transfer)
│   └── eval_rag_conflicts.sh
└── utils/
    ├── judge_math_answer_gpt.py  # GPT-4o-mini LLM-as-judge fallback
    ├── math_equal.py             # numerical answer equivalence (rule-based)
    ├── math_equivalence.py
    ├── metric_calc_rule.py
    └── prompts.py
```

## Benchmarks

| Suite       | Benchmarks                                                                |
|-------------|---------------------------------------------------------------------------|
| Math        | MATH-500, AIME-24/25, AMC-22/23, Minerva-Math, OlympiadBench              |
| Abstention  | AbstentionBench (20 datasets, paper §4.2; 3500 deterministic sub-samples) |
| RAG         | DRAGged-into-Conflicts (paper Table 4, zero-shot)                          |

## Dependencies

> **Note**: evaluation pins a *different* vLLM version than training. We
> recommend creating a separate conda env for evaluation.

```bash
# Default eval env (works for Qwen3 and Llama-3.2 checkpoints)
pip install "vllm>=0.8.5" "transformers>=4.57.0" tqdm openai httpx math-verify
```

### OLMo-3-32B evaluation env (separate)

The OLMo-3 family needs newer vLLM kernels. Create a dedicated env:

```bash
conda create -n cpt-eval-olmo3 python=3.10 -y
conda activate cpt-eval-olmo3
pip install "vllm>=0.12.0" "transformers>=4.57.0" tqdm openai httpx math-verify
```

## OpenAI / LLM-as-a-Judge

Both math and abstention evaluation use **GPT-4o-mini** as a fallback judge.
Set the credentials via environment variables before launching:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional override
```

## Datasets

Place the benchmark files under `eval/dataset/`. Datasets are released on
ModelScope:

```bash
pip install modelscope
modelscope download --dataset Tsinghuadhy/CPT-Eval-Math              --local_dir eval/dataset
modelscope download --dataset Tsinghuadhy/CPT-Eval-AbstentionBench   --local_dir eval/dataset/abstention_bench
modelscope download --dataset Tsinghuadhy/CPT-Eval-RAGConflicts      --local_dir eval/dataset/conflicts
```

Resulting layout:

```
eval/dataset/
├── math500/test.jsonl
├── aime24/test.jsonl
├── aime25/test.jsonl
├── amc22/test.jsonl
├── amc23/test.jsonl
├── minervamath/test.jsonl
├── olympiad/test.jsonl
├── abstention_bench/                  # 20 datasets + raw_data/subsampling-indices.json
└── conflicts/conflicts.jsonl          # DRAGged-into-Conflicts
```

## Run

### Math

```bash
cd eval/math
bash eval_math.sh Tsinghuadhy/CPT-RL-Qwen3-14B
```

### AbstentionBench (Normal-Prompt + Abstention-Prompt)

```bash
cd eval/abstention
bash eval_abstention_batch.sh Tsinghuadhy/CPT-RL-Qwen3-14B
```

### RAG-Conflicts (zero-shot transfer)

```bash
cd eval/rag
bash eval_rag_conflicts.sh Tsinghuadhy/CPT-RL-Qwen3-4B
```

Each script writes per-dataset metrics to `outputs/<task>/<model_name>/`.
