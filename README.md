<div align="center">

# Cognitive Pairwise Training (CPT)

### *Enhancing Model Metacognition via Cognitive Pairwise Training*

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-B31B1B.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-EA4335.svg)](./paper.pdf)
[![Status](https://img.shields.io/badge/Status-Preprint-743482.svg)]()
[![Scales](https://img.shields.io/badge/Models-3B%20→%2032B-743482.svg)]()
[![License](https://img.shields.io/badge/License-MIT-2E8B57.svg)]()

**Tsinghua University**

<sub><i>A pairwise mid-training stage that preserves abstention through RLVR.</i></sub>

<br/>

<table align="center" border="0" cellspacing="0" cellpadding="0">
<tr>
<td align="center" width="50%"><img src="figures/fig_a_padded.png" width="100%" alt="The abstention task"/></td>
<td align="center" width="50%"><img src="figures/fig_c.png" width="100%" alt="CPT pipeline overview"/></td>
</tr>
<tr>
<td align="center"><sub><b>The abstention task.</b></sub></td>
<td align="center"><sub><b>CPT Method illustration.</b></sub></td>
</tr>
</table>

</div>

---

## 🎯 Problem: RL Collapses Abstention

After RLVR, models tend to answer questions they previously declined. On Llama-3.2-3B, vanilla SFT+RL drops Abstention Recall by 17.2 pp under the Normal Prompt and 13.6 pp under the Abstention Prompt.

<div align="center">
  <img src="figures/readme_fig1b_rl_collapse.png" width="62%" alt="RL collapses abstention on Llama-3.2-3B"/>
  <br/>
  <sub><b>Figure 1(b).</b> Abstention Recall before vs. after Math-RL, Llama-3.2-3B.</sub>
</div>

| Mid-training (3B) | Normal F1 (pre → post-RL) | ΔF1 | Recall (pre → post) | ΔRecall |
|:--|:--:|:--:|:--:|:--:|
| Vanilla SFT       | 60.3 → 45.4 | −14.9 | 48.0 → 30.8 | −17.2 |
| **CPT (ours)**    | 61.4 → 56.5 | **−4.9**  | 50.0 → 42.6 | **−7.4** |

The same pattern holds at 4B, 8B, and 14B; at 14B, CPT is the only mid-training whose Normal F1 and Recall do not decrease after RL (ΔF1 = +0.6, ΔRecall = +0.4).

---

## 💡 The Idea — Pairwise Mid-Training

CPT supervises the model with pairwise comparisons of reasoning traces rather than point-wise scores. Given a problem $x$, its reference answer $a^*$, and two candidate traces $\tau^A, \tau^B$, the model is trained to output a short comparative analysis and a relative-quality label. This pairwise objective is inserted as a mid-training stage between the pretrained LLM and the standard Math-SFT → Math-RL recipe.

**Two views of CPT.** The training-recipe view shows *where* CPT plugs into the stack; the method view (Fig. 2 of the paper) shows *what* CPT actually optimizes — a pairwise reasoning-comparison objective built from multi-model rollouts, debiased pair construction, and self-consistent teacher labels.

<div align="center">
  <img src="figures/readme_pipeline.png" width="92%" alt="CPT training pipeline"/>
  <br/>
  <sub><b>Figure 2(i). Training pipeline.</b> CPT is a pairwise SFT mid-training stage inserted between the pretrained LLM and the Math-SFT → Math-RL recipe.</sub>
</div>

<br/>

<div align="center">
  <img src="figures/readme_method_overview.png" width="96%" alt="CPT method overview (paper Fig. 2)"/>
  <br/>
  <sub><b>(ii) Method overview (paper Fig. 2).</b> A difficulty-balanced problem pool feeds <i>multi-model rollouts</i> (Qwen3-4B / 8B / 14B Base), which are then assembled into <i>debiased trace pairs</i> (Intra-Model · Inter-Model · Small-correct vs. Large-wrong) and assigned <i>self-consistent teacher labels</i> (Qwen3-235B, 8 rounds × 4 axes). The resulting pairwise comparison task trains <i>f<sub>θ</sub></i> to internalize a reusable criterion for reasoning quality before downstream task-specific optimization.</sub>
</div>

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

## 🔬 Further Analyses

### 1. Trace Quality at Matched Correctness (14B, Qwen3-235B pairwise judge)

On 247 consensus non-tie pairs of CPT-RL vs. SFT-RL traces, a Qwen3-235B judge prefers CPT-RL 55.9% of the time overall. The gap is concentrated on the `both_wrong` slice (64.6%) and the hardest subset (AIME-25 ∩ `both_wrong`, 83.0%), indicating that when CPT-RL fails it tends to fail with more structured reasoning rather than fabricated derivations.

| Slice | $n$ | **Position-debiased win-rate (Ours)** |
|:--|:--:|:--:|
| All consensus non-tie pairs            | 247 | **55.9%** |
| `both_correct`                         | 144 | 49.9% |
| `both_wrong`                           | 103 | **64.6%** |
| AIME-25 & `both_wrong` (hardest slice) |  36 | **83.0%** |

<div align="center">
  <img src="figures/readme_case_study.png" width="92%" alt="Pairwise case study: rotated parabola"/>
  <br/>
  <sub><b>Case study (AIME-2025, both_wrong).</b> Both predictions are wrong (69 vs. 70; ground truth = 62). The judge prefers Ours 7/8 at <i>Very High</i> confidence: Ours derives the correct quartic structure; the baseline introduces a coefficient error and concludes by guessing.</sub>
</div>

### 2. Zero-shot Transfer to RAG (4B on *DRAGged-into-Conflicts*)

The 4B CPT+RL checkpoint is evaluated on RAG abstention without any RAG-specific training.

| Model (4B)        | Normal | Outdated-info | Conflict (abstain) | **All (best-of-8)** |
|:--|:--:|:--:|:--:|:--:|
| Base              | 68.4 | 51.2 | 42.1 | 66.2 |
| SFT+RL            | 76.7 | 57.3 | 31.6 | 71.6 |
| Abs-RL            | 75.5 | 57.5 | 39.0 | 73.4 |
| **CPT+RL (ours)** | **80.4** | **61.5** | 35.1 | **78.6** |

### 3. Closed-Loop Self-Distillation (32B replaces 235B teacher)

A 32B CPT checkpoint re-labels its own pairwise training data and is used as the teacher for a fresh CPT run. The resulting 32B model matches the original Qwen3-235B–taught checkpoint on math and Normal-F1, removing the dependency on a closed-source teacher.

| Teacher | Math Avg | Normal-F1 | Abs.-F1 |
|:--|:--:|:--:|:--:|
| Math-SFT only                | 68.9 | 64.8 | 68.7 |
| Qwen3-235B (closed)          | 69.1 | 65.9 | **73.6** |
| **32B self-distilled (ours)** | **69.2** | **66.2** | 72.1 |

### 4. CPT-Data Ablations — robust and redundant

Removing any single pair-construction strategy (T1 / T2 / T3) or the self-consistency filter degrades performance, but every ablation still outperforms vanilla SFT. The three pair strategies contribute non-overlapping signal; the self-consistency filter has the largest effect at smaller scales.

---

## 📁 Repository Structure

```
.
├── README.md                    # This file
├── figures/                     # README figures
```

LaTeX/arxiv source will be linked once the arXiv version is posted.

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
