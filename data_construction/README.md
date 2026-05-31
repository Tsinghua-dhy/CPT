# Data Construction

End-to-end pipeline that produces the CPT mid-training data
(paper §3.1; reproduces all four [public datasets](#released-artifacts)).

## Stages

```
data_construction/
├── 1_rollout/                # Stage 1: multi-model rollouts on a difficulty-balanced problem pool
│   ├── rollout.py
│   ├── rollout_qwen3_4b.sh
│   ├── rollout_qwen3_8b.sh
│   └── rollout_qwen3_14b.sh
├── 2_pair/                   # Stage 2: debiased pair construction (Intra / Inter / Counter-intuitive)
│   └── build_pairs.py
├── 3_judge/                  # Stage 3: self-consistent teacher labeling (Qwen3-235B, K=8, threshold=5)
│   ├── judge_pairs.py
│   └── run_judge_example.sh
└── 4_build_sft/              # Stage 4: consensus filter + highest-confidence expansion
    ├── split_consensus.py
    └── build_sft_dataset.py
```

Counts in the paper (Table 1):

| Stage | Output                                                              |
|-------|----------------------------------------------------------------------|
| 1     | 8,556 problems × {4B, 8B, 14B} multi-rollout traces                  |
| 2     | 90,970 trace pairs (Intra 39K + Inter 39K + Small-correct/Large-wrong 13K) |
| 3     | 77,657 consensus-judged pairs (consensus rate 85.37%)                |
| 4     | 70,352 SFT instances (10K sampled → 7× highest-confidence expansion) |

## Reproduce vs. just use

If you only want to **use** the CPT data, download from ModelScope and skip this
directory entirely:

| Dataset                                                                                               | Content              |
|-------------------------------------------------------------------------------------------------------|-----------------------|
| [Tsinghuadhy/CPT-Source-8556](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-Source-8556)         | Stage 1 problem pool  |
| [Tsinghuadhy/CPT-Pairs-90K](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-Pairs-90K)             | Stage 2 raw pairs     |
| [Tsinghuadhy/CPT-Pairs-Judged-77K](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-Pairs-Judged-77K) | Stage 3 consensus pairs |
| [Tsinghuadhy/CPT-SFT-70K](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-SFT-70K)                 | Stage 4 SFT split     |

## Run end-to-end

```bash
# Stage 1: rollouts (requires 8 x 80GB GPUs per size; takes a couple of hours each)
bash 1_rollout/rollout_qwen3_4b.sh
bash 1_rollout/rollout_qwen3_8b.sh
bash 1_rollout/rollout_qwen3_14b.sh

# Stage 2: pair construction (one CPU run)
python 2_pair/build_pairs.py

# Stage 3: 235B self-consistency judging (8x80GB; ~5 days per split)
bash 3_judge/run_judge_example.sh

# Stage 4: consensus filter + SFT expansion
python 4_build_sft/split_consensus.py
python 4_build_sft/build_sft_dataset.py
```

Each script reads from the previous stage's output and the default paths are
relative to this directory.

## Notes

- Rollout scoring uses **math_verify** (rule-based) plus a GPT-4o-mini
  fallback for semantic-equivalence checks. Set `OPENAI_API_KEY` /
  `OPENAI_BASE_URL` in your env before running stages 1 or 4.
- The Qwen3-235B teacher in Stage 3 is the standard
  `Qwen3-235B-A22B-Instruct-2507` checkpoint. We also support a
  self-distilled 32B teacher (paper §6.3); swap the `model_path` in
  `judge_pairs.py`.
