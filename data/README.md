# Data

Train + eval data used by the [CPT](https://github.com/Tsinghua-dhy/CPT) pipeline.

## What ships in this repo (small splits)

| Folder                       | Used by                                    | Size |
|------------------------------|--------------------------------------------|------|
| `math_sft/`                  | `train/sft/run_math_sft_qwen3_14b.sh` (Base SFT warm-up) | 32 MB |
| `math_sft_instruct/`         | `train/sft/run_math_sft_llama_3b_instruct.sh`            | 32 MB |
| `math_rl/`                   | `train/rl/run_math_rl_qwen3_base.sh` (Math-RL train split) | 9 MB  |
| `math_rl_instruct/`          | `train/rl/run_math_rl_llama_3b_instruct.sh`              | 9 MB  |
| `math_benchmarks/`           | RL eval set (MATH-500, AIME-24/25, AMC-22/23, Minerva, Olympiad)  | 0.3 MB |
| `math_benchmarks_instruct/`  | Same as above, instruct-formatted          | 0.6 MB |

`cpt_sft/` (the 70K mid-training split used by `train/sft/run_cpt_*.sh`) is too
large for this repo: download it from ModelScope (see below) and place it under
`data/cpt_sft/`.

## What lives on ModelScope

### CPT cognitive data (paper §3.1)

| Repo                                                                                                  | Stage    | Purpose                              |
|-------------------------------------------------------------------------------------------------------|----------|--------------------------------------|
| [`Tsinghuadhy/CPT-Source-8556`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-Source-8556)         | 1        | Difficulty-balanced problem pool     |
| [`Tsinghuadhy/CPT-Pairs-90K`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-Pairs-90K)             | 2        | Trace pairs before judging           |
| [`Tsinghuadhy/CPT-Pairs-Judged-77K`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-Pairs-Judged-77K) | 3        | Consensus-judged pairs (slim) |
| [`Tsinghuadhy/CPT-SFT-70K`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-SFT-70K)                 | 4        | Final mid-training SFT data          |

### Baseline training data (paper §4.1, §4.3)

| Repo                                                                                                                | Used by                                |
|---------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| [`Tsinghuadhy/CPT-TrainingData-SFT80K`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-TrainingData-SFT80K)         | `train/sft/run_sft80k_qwen3_4b.sh`      |
| [`Tsinghuadhy/CPT-TrainingData-AbstentionSFT`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-TrainingData-AbstentionSFT) | `train/sft/run_abs_sft_qwen3_4b.sh`     |
| [`Tsinghuadhy/CPT-TrainingData-DPOPairs`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-TrainingData-DPOPairs)     | `train/dpo/run_dpo_qwen3_8b.sh`         |

### One-line download

```bash
pip install modelscope
modelscope download --dataset Tsinghuadhy/CPT-SFT-70K --local_dir data/cpt_sft
modelscope download --dataset Tsinghuadhy/CPT-TrainingData-SFT80K --local_dir data/math_sft_80k
modelscope download --dataset Tsinghuadhy/CPT-TrainingData-AbstentionSFT --local_dir data/abstention_sft
modelscope download --dataset Tsinghuadhy/CPT-TrainingData-DPOPairs --local_dir data/dpo_baseline
```

## Evaluation datasets

`eval/dataset/` (download from ModelScope):

| Repo                                                                                                              | Content                              |
|-------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| [`Tsinghuadhy/CPT-Eval-Math`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-Eval-Math)                       | MATH-500, AIME-24/25, AMC-22/23, Minerva, Olympiad |
| [`Tsinghuadhy/CPT-Eval-AbstentionBench`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-Eval-AbstentionBench) | AbstentionBench (20 datasets + sub-sampling indices) |
| [`Tsinghuadhy/CPT-Eval-RAGConflicts`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-Eval-RAGConflicts)       | DRAGged-into-Conflicts                |

```bash
modelscope download --dataset Tsinghuadhy/CPT-Eval-Math            --local_dir eval/dataset
modelscope download --dataset Tsinghuadhy/CPT-Eval-AbstentionBench --local_dir eval/dataset/abstention_bench
modelscope download --dataset Tsinghuadhy/CPT-Eval-RAGConflicts    --local_dir eval/dataset/conflicts
```
