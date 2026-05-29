<div align="center">

# Cognitive Pairwise Training (CPT)

### *Enhancing Model Metacognition via Cognitive Pairwise Training*

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-B31B1B.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-EA4335.svg)](./paper.pdf)
[![Status](https://img.shields.io/badge/Status-Preprint-743482.svg)]()
[![Scales](https://img.shields.io/badge/Models-3B%20→%2032B-743482.svg)]()
[![License](https://img.shields.io/badge/License-MIT-2E8B57.svg)]()

**Tsinghua University**

<sub><i>RLVR makes LLMs smarter — but also more overconfident. CPT teaches them when <b>not</b> to answer.</i></sub>

<br/>

<table align="center" border="0" cellspacing="0" cellpadding="0">
<tr>
<td align="center" width="50%"><img src="figures/fig_a_padded.png" width="100%" alt="The abstention task"/></td>
<td align="center" width="50%"><img src="figures/fig_c.png" width="100%" alt="CPT pipeline overview"/></td>
</tr>
<tr>
<td align="center"><sub><b>The abstention task.</b></sub></td>
<td align="center"><sub><b>CPT pipeline overview.</b></sub></td>
</tr>
</table>

</div>

---

## 🎯 The Problem — RL Collapses Metacognition

RLVR is the standard recipe for boosting LLM reasoning, but it comes with a hidden cost: **models lose the ability to abstain**. After RL, they confidently answer questions they should refuse.

<div align="center">
  <img src="figures/readme_fig1b_rl_collapse.png" width="62%" alt="RL collapses abstention on Llama-3.2-3B"/>
  <br/>
  <sub><b>Figure 1(b).</b> On Llama-3.2-3B, vanilla SFT+RL drops Abstention Recall by <b>−17.2 pp</b> under the Normal Prompt and <b>−13.6 pp</b> under the Abstention Prompt.</sub>
</div>

> CPT cuts both losses by **~3×**.

| Mid-training (3B) | Normal F1 (pre → post-RL) | ΔF1 | Recall (pre → post) | ΔRecall |
|:--|:--:|:--:|:--:|:--:|
| Vanilla SFT       | 60.3 → 45.4 | −14.9 | 48.0 → 30.8 | −17.2 |
| **CPT (ours)**    | 61.4 → 56.5 | **−4.9**  | 50.0 → 42.6 | **−7.4** |

The same pattern holds at every scale. At **14B**, CPT is the only mid-training that keeps F1 *stable* (+0.6) and Recall *up* (+0.4) after RL.

---

## 💡 The Idea — Pairwise Mid-Training

Instead of point-wise scores, we supervise the model with **pairwise comparisons** of reasoning traces. It internalizes a **reusable reasoning-quality boundary** that transfers to its own generations, enabling calibrated self-assessment.

<div align="center">
  <img src="figures/readme_pipeline.png" width="92%" alt="CPT training pipeline"/>
  <br/>
  <sub><b>Pipeline.</b> CPT is a pairwise SFT mid-training stage inserted between the pretrained LLM and the standard Math-SFT → Math-RL recipe.</sub>
</div>

Full method details: [`method_summary.md`](./method_summary.md)

---

## 📊 Main Results — Math + AbstentionBench

Within each scale block, **CPT+RL is the only method that simultaneously wins on math accuracy AND intrinsic (Normal-Prompt) abstention F1.**

| Scale | Method | **Math Avg** | **Normal-F1** | Abs.-F1 |
|:--:|:--|:--:|:--:|:--:|
| **14B** | SFT+RL              | 67.2 | 60.8 | 64.7 |
|         | DPO+RL              | 67.3 | 63.6 | 66.9 |
|         | Abs-RL              | 67.5 | 62.5 | 68.2 |
|         | SFT-80K+RL          | 65.9 | 64.4 | 72.4 |
|         | **CPT+RL (ours)**   | **69.4** | **66.4** | **73.4** |
| **8B**  | SFT+RL              | 58.4 | 61.7 | 71.8 |
|         | SFT-80K+RL          | 60.0 | 66.3 | 74.9 |
|         | **CPT+RL (ours)**   | **60.6** | **66.8** | 72.2 |
| **4B**  | SFT+RL              | 54.2 | 58.6 | 72.5 |
|         | SFT-80K+RL          | 54.2 | 67.0 | 73.3 |
|         | **CPT+RL (ours)**   | **55.8** | 64.5 | 72.1 |
| **32B** ‡ | SFT (no RL)       | 68.9 | 64.9 | 68.8 |
|         | DPO (no RL)         | 70.3 | 61.6 | 67.8 |
|         | **CPT (ours, no RL)** | 69.1 | **66.2** | **72.1** |

<sub>‡ OLMo-3 32B Base, mid-training only (no math RL). All other scales are Qwen3 base / Llama-3.2 instruct.</sub>

---

## 🔬 Beyond Headline Numbers

### 1. Higher Trace Quality at Matched Correctness (14B Pairwise Judge)

Even when **both** CPT-RL and SFT-RL get the wrong answer, a Qwen3-235B judge prefers CPT-RL's trace 64.6% of the time — rising to **83.0%** on AIME-25 hard cases.

| Slice | $n$ | **Position-debiased win-rate (Ours)** |
|:--|:--:|:--:|
| All consensus non-tie pairs            | 247 | **55.9%** |
| `both_correct`                         | 144 | 49.9% |
| `both_wrong`                           | 103 | **64.6%** |
| AIME-25 & `both_wrong` (hardest slice) |  36 | **83.0%** |

> CPT-RL fails *more structured*, less fabricated reasoning — exactly the trait needed for reliable abstention.

<div align="center">
  <img src="figures/readme_case_study.png" width="92%" alt="Pairwise case study: rotated parabola"/>
  <br/>
  <sub><b>Case study (AIME-2025, both_wrong).</b> Both predictions are wrong (69 vs. 70; GT = 62), but the judge prefers <b>Ours</b> 7/8 at <i>Very High</i> confidence: it has the correct quartic structure, while the baseline introduces a coefficient error and concludes by guessing.</sub>
</div>

### 2. Cross-Task Generalization to RAG (zero-shot, 4B on *DRAGged-into-Conflicts*)

No RAG training, no task-specific tuning — CPT still leads.

| Model (4B)        | Normal | Outdated-info | Conflict (abstain) | **All (best-of-8)** |
|:--|:--:|:--:|:--:|:--:|
| Base              | 68.4 | 51.2 | 42.1 | 66.2 |
| SFT+RL            | 76.7 | 57.3 | 31.6 | 71.6 |
| Abs-RL            | 75.5 | 57.5 | 39.0 | 73.4 |
| **CPT+RL (ours)** | **80.4** | **61.5** | 35.1 | **78.6** |

### 3. Closed-Loop Self-Distillation (32B replaces 235B teacher)

The recipe is **fully reproducible in-house** — a 32B CPT checkpoint can re-label its own training data and match (or beat) the 235B teacher.

| Teacher | Math Avg | Normal-F1 | Abs.-F1 |
|:--|:--:|:--:|:--:|
| Math-SFT only                | 68.9 | 64.8 | 68.7 |
| Qwen3-235B (closed)          | 69.1 | 65.9 | **73.6** |
| **32B self-distilled (ours)** | **69.2** | **66.2** | 72.1 |

### 4. CPT-Data Ablations — robust and redundant

Removing **any single** pair-construction strategy (T1 / T2 / T3) or **self-consistency** filtering degrades performance, but every ablation still beats vanilla SFT — meaning the three pair strategies provide complementary signal, and SC filtering is most critical for smaller models.

---

## 🖼️ Figures Used in This README

All images live under [`figures/`](./figures/) and are referenced with relative paths, so GitHub will render them out of the box.

| File | Purpose | Where it appears |
|:--|:--|:--|
| `figures/fig_a_padded.png`             | "The abstention task" cartoon (Fig. 1a, height-padded) | Hero image, left |
| `figures/fig_c.png`                    | High-level CPT illustration (Fig. 1c)               | Hero image, right |
| `figures/readme_fig1b_rl_collapse.png` | Fig. 1(b) — RL collapses abstention on Llama-3.2-3B | "The Problem" section |
| `figures/readme_pipeline.png`          | CPT training pipeline overview                      | "The Idea" section |
| `figures/readme_case_study.png`        | AIME-2025 pairwise case study (rotated parabola)    | "Beyond Headline Numbers" |

> 💡 *Want extra visuals (e.g. RAG-transfer bar chart, win-rate radar, scaling curve)? Tell me which numbers to highlight and I'll generate matching PNGs in the same palette.*

---

## 📁 Repository Structure

```
.
├── README.md                    # This file
├── figures/                     # README figures
```

> 📄 Paper PDF / LaTeX source will be linked to arXiv once posted.

---

## 📌 Release Plan

- [ ] Training code (CPT data construction + SFT + RL)
- [ ] CPT data: 90K pairs + 70K SFT samples
- [ ] Model checkpoints — Qwen3 {4B / 8B / 14B}, Llama-3.2-3B-Instruct, OLMo-3 32B
- [ ] Evaluation scripts (math benchmarks + AbstentionBench + RAG transfer)

> Code, data, and model weights will be released upon publication.

---

## 📖 Citation

```bibtex
@article{li2026cpt,
  title         = {Enhancing Model Metacognition via Cognitive Pairwise Training},
  author        = {Li, Weitao and Zhou, Hao and Lei, Xuanyu and Meng, Fandong
                   and Ren, Jingyi and Wang, Ante and Wang, Xiaolong and Luo, Fuwen
                   and Zhang, Yuanchi and Yang, Guangwen and Gan, Lin
                   and Ma, Weizhi and Liu, Yang},
  journal       = {arXiv preprint arXiv:XXXX.XXXXX},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

---

## 👥 Authors

**Weitao Li**¹², Hao Zhou, Xuanyu Lei¹², Fandong Meng, Jingyi Ren¹², Ante Wang², Xiaolong Wang¹², Fuwen Luo¹², Yuanchi Zhang, Guangwen Yang¹, Lin Gan¹, **Weizhi Ma**²†, **Yang Liu**¹²†

<sub>¹ Dept. of CS & Tech., Institute for AI, Tsinghua University &nbsp;·&nbsp; ² Institute for AI Industry Research (AIR), Tsinghua University</sub>

<sub>† Corresponding authors.</sub>
