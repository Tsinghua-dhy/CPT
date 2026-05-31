# Training

End-to-end recipes for **Cognitive Pairwise Training (CPT)** and all baselines.
We use [verl](https://github.com/volcengine/verl) for both SFT and RL. A
sanitized snapshot of the verl tree we trained against is shipped at
[`../verl/`](../verl/).

## Install

```bash
# 1) Install the pinned verl snapshot
pip install -e ../verl

# 2) Install vLLM. GRPO training requires 0.8.5 (verl + GRPO rollouts are
#    pinned against this version).
pip install vllm==0.8.5

# 3) Transformers
pip install "transformers>=4.57.0"     # required for OLMo-3 base models
```

> Notes
> - `verl` is pinned to **0.5.0** (the snapshot in `../verl/`).
> - For OLMo-3 mid-training the same train environment works because
>   transformers ≥ 4.57 already ships the OLMo-3 model class.
> - **Evaluation** uses a *different* environment (vLLM 0.12.0). See
>   [`../eval/README.md`](../eval/README.md).

## Layout

```
train/
├── common/
│   ├── distributed_env.sh    # multi-node env (NCCL/IB) – source from every runner
│   └── ray_bootstrap.sh      # Ray head/worker bootstrap – source from RL runners
├── sft/
│   ├── _runner.sh                            # shared verl SFT launcher
│   ├── run_cpt_qwen3_4b.sh                   # CPT mid-training (Qwen3-4B-Base)
│   ├── run_cpt_qwen3_14b.sh                  # CPT mid-training (Qwen3-14B-Base, paper main)
│   ├── run_cpt_olmo3_32b.sh                  # CPT mid-training (OLMo-3-32B-Base, no RL after)
│   ├── run_math_sft_qwen3_14b.sh             # Math-SFT warm-up (SFT+RL baseline, Qwen3-14B)
│   ├── run_math_sft_llama_3b_instruct.sh     # Math-SFT warm-up (Llama-3.2-3B-Instruct)
│   ├── run_sft80k_qwen3_4b.sh                # SFT-80K volume-matched baseline (Qwen3-4B)
│   └── run_abs_sft_qwen3_4b.sh               # Abs-SFT warm-up (Abs-RL baseline, Qwen3-4B)
├── dpo/
│   └── run_dpo_qwen3_8b.sh                   # DPO+RL baseline (Qwen3-8B)
└── rl/
    ├── _runner.sh                            # shared verl GRPO launcher
    ├── run_math_rl_qwen3_base.sh             # Math-RL on Qwen3-Base family
    └── run_math_rl_llama_3b_instruct.sh      # Math-RL on Llama-3.2-3B-Instruct
```

## Multi-node setup

Each runner exports three env vars before sourcing `common/distributed_env.sh`:

```bash
export MASTER_NODE="<rank-0 reachable IP>"
export MASTER_PORT=23457
export NODE_LOCAL_IPS="<ip-0> <ip-1> ... <ip-N>"   # space-separated
```

The current host's local IP must appear in `NODE_LOCAL_IPS`; its index becomes
`NODE_RANK`. Single-node runs work with the defaults (`127.0.0.1`).

## End-to-end pipeline

The full **CPT** pipeline (paper §4.1) is three stages:

```
Base
  ├─ [stage 1] CPT mid-training        # sft/run_cpt_<size>.sh
  ├─ [stage 2] Math-SFT warm-up        # sft/run_math_sft_*.sh
  └─ [stage 3] Math-RL (GRPO)          # rl/run_math_rl_*.sh   (set MODEL_PATH = stage-2 ckpt)
```

Each baseline replaces one stage:

| Baseline       | Stage 1                                     |
|----------------|---------------------------------------------|
| `SFT+RL`       | *skipped*                                   |
| `SFT-80K+RL`   | `sft/run_sft80k_qwen3_4b.sh`                |
| `DPO+RL`       | `sft/run_math_sft_*.sh` + `dpo/run_dpo_*.sh`|
| `Abs-RL`       | `sft/run_math_sft_*.sh` + Math-RL + `sft/run_abs_sft_qwen3_4b.sh` + Math-RL |
| `CPT+RL`(ours) | `sft/run_cpt_*.sh`                          |

For OLMo-3-32B we run mid-training only (no Math-RL after).

## Data

Train / eval splits expected by these scripts:

| Dir                                | Source                                                                                                  |
|------------------------------------|---------------------------------------------------------------------------------------------------------|
| `data/cpt_sft/`                    | [`CPT-SFT-70K`](https://www.modelscope.cn/datasets/Tsinghuadhy/CPT-SFT-70K)                            |
| `data/math_sft/`                   | Math-SFT 10K warm-up (see `data_construction/4_build_sft/`)                                            |
| `data/math_sft_80k/`               | Volume-matched 80K answerable-math SFT (see `data_construction/4_build_sft/`)                          |
| `data/abstention_sft/`             | Built from public abstention data; build script in `data_construction/4_build_sft/`                    |
| `data/math_rl/`                    | DAPO-Math RL split                                                                                      |
| `data/math_benchmarks/`            | MATH-500 / AIME-24/25 / AMC-22/23 / Minerva / OlympiadBench (`eval/math/`)                              |

Drop the parquet files into the corresponding folder before launching a script,
or override `TRAIN_FILE` / `VAL_FILE` via environment variables.
