"""
Evaluate three task types in conflicts.jsonl using LLM-as-a-judge: 

task split (by conflict_type field): 
  - answerable_normal:        ["No conflict", "Conflict due to misinformation"]
        → model should answer correctly; the document is trustworthy, can combine with world knowledge
  - answerable_outdated:      ["Conflict due to outdated information"]
        → model should detect outdated-info conflicts and pick the most up-to-date answer
  - abstain (conflict / no consensus answer): ["Complementary information",
                                "Conflicting opinions and research outcomes"]
        → model should abstain and output \\boxed{I don't know}

Metrics: 
  - abstainclass (abstain): scored by whether the model abstains; abstaining counts as correct
  - the other two task types: LLM-as-a-judge compares pred_ans against correct_answer
  - overall acc         : aggregated over the three task types

report per-task accuracy and the overall accuracy.
"""

import argparse
from collections import defaultdict
import copy
import json
import os
import re
import sys
import time

from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Make eval/utils/ importable so we can reuse `metric_calc_rule.py`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

from metric_calc_rule import (
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
    GPT4omini_batch_request,
    get_judge_prompt,
)


# ============================================================================
#  1. task classification: conflict_type -> task_type (3 classes)
# ============================================================================

ANSWER_NORMAL_TYPES = {"No conflict", "Conflict due to misinformation"}
ANSWER_OUTDATED_TYPES = {"Conflict due to outdated information"}
ABSTAIN_TYPES = {
    "Complementary information",
    "Conflicting opinions and research outcomes",
}


def get_task_type(conflict_type: str) -> str:
    """Map the 5 conflict_type values down to the 3 task labels."""
    if conflict_type in ANSWER_OUTDATED_TYPES:
        return "answerable_outdated"
    if conflict_type in ABSTAIN_TYPES:
        return "abstain"
    if conflict_type in ANSWER_NORMAL_TYPES:
        return "answerable_normal"
    return "answerable_normal"  # fallback


TASK_TYPES_ORDER = ["answerable_normal", "answerable_outdated", "abstain"]


# ============================================================================
#  2. parameter
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=100000)
    parser.add_argument(
        "--data_file",
        type=str,
        default="./conflicts.jsonl",
    )
    parser.add_argument("--rollout_num", type=int, default=1, help="Number of rollouts per question")
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--output_dir", type=str, default="../outputs/rag_conflicts")
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Force using chat template. If not set, auto-detect based on model name.",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional tag for output file name")
    return parser.parse_args()


def is_instruct_model(model_name: str) -> bool:
    model_name_lower = (model_name or "").lower()
    instruct_keywords = ["instruct", "distill", "qwen2.5", "chat", "it"]
    return any(k in model_name_lower for k in instruct_keywords)


# ============================================================================
#  3. Prompt construction (the model itself recognizes the three task types)
# ============================================================================

USER_PROMPT = """Answer the following question. You are given reference documents and may also use your own knowledge.

Decision rule (apply IN ORDER):
  1. Look at the question and references. Ask yourself: "Across the references and my own knowledge, is there ONE single answer that essentially everyone would agree on?"
       - If different references answer about different countries / eras / persons / contexts because the question is ambiguous, the answer is NO.
       - If the topic is debated, opinion-based, or unresolved in research, the answer is NO.
  2. If NO -> you MUST abstain. Output exactly \\boxed{{I don't know}}.
  3. If YES -> answer it. (If some references are outdated, pick the most up-to-date one.)

When in doubt, prefer abstaining.

Give your reasoning first, then put a SHORT answer inside \\boxed{{...}}: an entity, a number, a short phrase, or exactly "I don't know". Do NOT put your reasoning inside the box.
  Examples: \\boxed{{Operation Market Garden}} | \\boxed{{1964}} | \\boxed{{Euro}} | \\boxed{{I don't know}}

Question: {question}

References:
{reference_text}
"""


def process_text(example, tokenizer, use_chat_template: bool = False):
    """Build the prompt. Base models reuse the training User:/Assistant: shell."""
    question_raw = example["question"]

    reference_parts = []
    for i, doc in enumerate(example.get("search_results", [])[:10]):
        short_text = doc.get("short_text") or doc.get("snippet") or ""
        if short_text:
            reference_parts.append(f"[{i + 1}] {short_text}")
    reference_text = "\n\n".join(reference_parts) if reference_parts else "(no reference document)"

    user_prompt = USER_PROMPT.format(question=question_raw, reference_text=reference_text)

    prompt = None
    if use_chat_template:
        try:
            prompt_chat = [{"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(
                prompt_chat, add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            print(f"[WARN] apply_chat_template failed, fallback to base prompt. Error: {e}")
            prompt = None

    if prompt is None:
        # base model: keep the User/Assistant scaffold
        prompt = (
            "The User asks a question, and the Assistant solves it.\n"
            "The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.\n"
            "The final answer should be enclosed within \\boxed{}.\n\n"
            "User:\n"
            f"{user_prompt}\n"
            "Assistant:"
        )

    example["chat_prompt"] = prompt
    return example


# ============================================================================
#  4. Answer extraction + abstention detection
# ============================================================================

def extract_boxed_content(text: str):
    """Extract every \\boxed{...} body from the string (handles nested braces)."""
    pattern = r"\\boxed\{"
    matches = []
    for match in re.finditer(pattern, text):
        start_pos = match.end() - 1  # points at '{'
        brace_count = 0
        content_start = start_pos + 1
        for i in range(start_pos, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    matches.append(text[content_start:i])
                    break
    return matches


def extract_answer_rag(s: str) -> str:
    """Extract the final answer: prefer the last \\boxed{}, then <answer>...</answer>."""
    all_matches = []
    all_matches.extend(extract_boxed_content(s))
    answer_pattern = r"<answer>(.*?)</answer>"
    all_matches.extend(re.findall(answer_pattern, s, re.DOTALL | re.IGNORECASE))
    return all_matches[-1].strip() if all_matches else ""


# Abstention keywords (low threshold, robust matching) 
ABSTAIN_PATTERNS = [
    r"\bi\s*don'?t\s*know\b",
    r"\bi\s*do\s*not\s*know\b",
    r"\bdon'?t\s*know\b",
    r"\bidk\b",
    r"\bunknown\b",
    r"\bno\s*(single|definitive|clear|consensus|agreed)\b",
    r"\bnot\s*answerable\b",
    r"\bcannot\s*(be\s*)?answer(ed)?\b",
    r"\bcan'?t\s*(be\s*)?answer(ed)?\b",
    r"\bno\s*(consensus|agreement)\b",
    r"\binsufficient\s*information\b",
    r"\bunable\s*to\s*answer\b",
    r"\bnot\s*known\b",
    r"\bn/?a\b",
    r"\babstain\b",
    r"\brefuse\s*to\s*answer\b",
]
ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)


def is_abstain(pred_ans: str) -> bool:
    """Check whether pred_ans is an abstention (e.g. "I don't know")."""
    if not pred_ans:
        # No \boxed{} answer: do NOT count as abstention (avoid scoring empty answers as correct) 
        return False
    return bool(ABSTAIN_RE.search(pred_ans))


# ============================================================================
#  5. Batched LLM-as-judge evaluation (answerable items only) 
# ============================================================================

def llm_judge_batch(eval_items):
    """
    Run batched LLM-as-judge on answerable samples. 

    eval_items: list of dict, each containing {"question", "answer" (gold), "pred_ans"}
    return:     list of dict {"score": 0/1, "raw_response": str, "reason": str}
                output order matches eval_items
    """
    if not eval_items:
        return []

    # Short-circuit: items with F1==1.0 are scored 1.0 directly
    prompts = []
    metadata = []  # (eval_idx, gold_idx)
    short_circuit = {}  # eval_idx -> score

    for idx, item in enumerate(eval_items):
        pred = item["pred_ans"] or ""
        gold = item["answer"]
        gold_list = gold if isinstance(gold, list) else [gold]
        gold_list = [g for g in gold_list if isinstance(g, str) and g.strip()]
        if not gold_list:
            # No valid gold answer -> cannot score; treat as 0
            short_circuit[idx] = 0.0
            continue
        # F1 short-circuit
        f1, _, _ = metric_max_over_ground_truths(f1_score, pred, gold_list)
        if f1 >= 1.0:
            short_circuit[idx] = 1.0
            continue
        for gi, g in enumerate(gold_list):
            prompts.append(get_judge_prompt(item["question"], g, pred))
            metadata.append((idx, gi))

    print(
        f"  [LLM-Judge] total={len(eval_items)}, "
        f"f1==1.0 short-circuit={len(short_circuit)}, pending LLM-judge prompts={len(prompts)}"
    )

    raw_results = GPT4omini_batch_request(prompts) if prompts else []

    grouped = defaultdict(list)  # eval_idx -> [(score, raw, reason), ...]
    for i, raw in enumerate(raw_results):
        eval_idx, _gi = metadata[i]
        rs = (raw or "").lower().strip()
        if "incorrect" in rs or rs == "false":
            score = 0
        elif "judgment: correct" in rs or rs == "correct" or rs == "true":
            score = 1
        else:
            score = 0
        # reason
        try:
            reason = (raw or "").split("Reason:", 1)[1].split("Judgment:", 1)[0].strip()
        except Exception:
            reason = raw or ""
        grouped[eval_idx].append((score, raw, reason))

    final = []
    for idx in range(len(eval_items)):
        if idx in short_circuit:
            final.append(
                {
                    "score": float(short_circuit[idx]),
                    "raw_response": "Skipped (F1==1.0 or no valid gold)",
                    "reason": "short-circuit",
                }
            )
        else:
            ress = grouped.get(idx, [])
            if not ress:
                final.append({"score": 0.0, "raw_response": "LLM eval failed", "reason": ""})
            else:
                # multiple gold answers: take the max
                best = max(ress, key=lambda x: x[0])
                final.append(
                    {
                        "score": float(best[0]),
                        "raw_response": best[1],
                        "reason": best[2],
                    }
                )
    return final


# ============================================================================
#  6. main
# ============================================================================

def main():
    print("=" * 30 + " RAG Conflicts Evaluation " + "=" * 30)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    t_start = time.time()

    model_path = args.model_path
    model_short_name = (
        model_path.rstrip("/").split("/")[-2].lower()
        + "_"
        + model_path.rstrip("/").split("/")[-1].lower()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_gpus = torch.cuda.device_count()

    print(f"Loading model: {model_path}")
    print(f"Using {num_gpus} GPUs")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=args.gpu_memory_rate,
        trust_remote_code=True,
    )

    use_chat_template = args.use_chat_template or is_instruct_model(model_path)
    print(
        f"Use chat template: {use_chat_template} "
        f"(manual: {args.use_chat_template}, auto: {is_instruct_model(model_path)})"
    )

    rollout_num = args.rollout_num
    print(f"Rollout number: {rollout_num}")

    # ==== Load data (keep all 5 conflict_types, group into 3 task types) ====
    print(f"\nLoading data from: {args.data_file}")

    data_ori_all = []
    with open(args.data_file, "r") as f:
        for i, line in enumerate(f):
            if args.start_sample <= i < args.end_sample:
                obj = json.loads(line)
                obj["task_type"] = get_task_type(obj.get("conflict_type", ""))
                data_ori_all.append(obj)
            if i >= args.end_sample - 1:
                break

    print(f"Total samples loaded: {len(data_ori_all)}")
    task_dist = defaultdict(int)
    conflict_dist = defaultdict(int)
    for it in data_ori_all:
        task_dist[it["task_type"]] += 1
        conflict_dist[it.get("conflict_type", "Unknown")] += 1
    print("Task type distribution:")
    for k in TASK_TYPES_ORDER:
        print(f"  {k}: {task_dist[k]}")
    print("Underlying conflict_type distribution:")
    for k, v in conflict_dist.items():
        print(f"  {k}: {v}")

    t_run = time.time()

    # ---- generation ----
    print(f"Generating {rollout_num} rollouts × {len(data_ori_all)} questions ...")
    data = [copy.deepcopy(it) for it in data_ori_all]
    for it in data:
        process_text(it, tokenizer, use_chat_template=use_chat_template)

    prompts = []
    prompt_to_qr = []  # (q_idx, rollout_idx)
    for qi, it in enumerate(data):
        for r in range(rollout_num):
            prompts.append(it["chat_prompt"])
            prompt_to_qr.append((qi, r))

    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    print(f"Total prompts: {len(prompts)}")
    outputs = llm.generate(prompts, sampling_params)

    rollouts = defaultdict(list)
    for out_idx, out in enumerate(outputs):
        q_idx, r_idx = prompt_to_qr[out_idx]
        gen_text = out.outputs[0].text.strip()
        pred = extract_answer_rag(gen_text)
        rollouts[q_idx].append(
            {
                "rollout_idx": r_idx,
                "pred_ans": pred,
                "gen_text_store": gen_text,
            }
        )

    # ---- Eval: abstention via rules; everything else via LLM-judge ----
    print("\n== Evaluating ==")
    eval_items = []
    eval_map = []  # (q_idx, r_idx)

    for q_idx, item in enumerate(data):
        ttype = item["task_type"]
        for r in rollouts[q_idx]:
            r["abstained"] = is_abstain(r["pred_ans"])
            if ttype == "abstain":
                r["correct"] = bool(r["abstained"])
                r["judge_score"] = float(r["correct"])
                r["judge_raw"] = (
                    "abstain rule: abstained -> correct"
                    if r["abstained"]
                    else "abstain rule: not abstained -> wrong"
                )
                r["judge_reason"] = ""
            else:
                if r["abstained"]:
                    r["correct"] = False
                    r["judge_score"] = 0.0
                    r["judge_raw"] = "answerable rule: model abstained -> wrong"
                    r["judge_reason"] = ""
                else:
                    eval_items.append(
                        {
                            "question": item["question"],
                            "answer": item.get("correct_answer", ""),
                            "pred_ans": r["pred_ans"],
                        }
                    )
                    eval_map.append((q_idx, r["rollout_idx"]))

    print(f"  abstain items judged directly; {len(eval_items)} answerable rollouts dispatched to LLM-as-judge")
    judge_results = llm_judge_batch(eval_items)
    for i, jr in enumerate(judge_results):
        q_idx, r_idx = eval_map[i]
        for r in rollouts[q_idx]:
            if r["rollout_idx"] == r_idx:
                r["correct"] = bool(jr["score"] >= 1.0)
                r["judge_score"] = float(jr["score"])
                r["judge_raw"] = jr["raw_response"]
                r["judge_reason"] = jr["reason"]
                break

    # ---- Aggregate metrics ----
    by_task = defaultdict(
        lambda: {"avg": [], "majority": 0, "best": 0, "count": 0}
    )
    overall_avg, overall_maj, overall_best, n_q = 0.0, 0, 0, 0

    detailed = []
    for q_idx, item in enumerate(data):
        rs = rollouts[q_idx]
        scores = [r["judge_score"] for r in rs]
        avg = float(np.mean(scores)) if scores else 0.0
        majority = 1 if avg > 0.5 else 0
        best = 1 if (max(scores) if scores else 0) >= 1.0 else 0

        ttype = item["task_type"]
        by_task[ttype]["avg"].append(avg)
        by_task[ttype]["majority"] += majority
        by_task[ttype]["best"] += best
        by_task[ttype]["count"] += 1
        overall_avg += avg
        overall_maj += majority
        overall_best += best
        n_q += 1

        for r in rs:
            detailed.append(
                {
                    "question_idx": q_idx,
                    "rollout_idx": r["rollout_idx"],
                    "question": item["question"],
                    "correct_answer": item.get("correct_answer", ""),
                    "conflict_type": item.get("conflict_type", ""),
                    "task_type": ttype,
                    "pred_ans": r["pred_ans"],
                    "abstained": r["abstained"],
                    "correct": r["correct"],
                    "judge_score": r["judge_score"],
                    "judge_raw": r["judge_raw"],
                    "judge_reason": r["judge_reason"],
                    "gen_text_store": r["gen_text_store"],
                }
            )

    def _safe_div(a, b):
        return float(a) / b if b else 0.0

    by_task_results = {}
    for k in TASK_TYPES_ORDER:
        v = by_task[k]
        by_task_results[k] = {
            "count": v["count"],
            "avg_acc": _safe_div(sum(v["avg"]), v["count"]),
            "majority_acc": _safe_div(v["majority"], v["count"]),
            "best_acc": _safe_div(v["best"], v["count"]),
        }

    overall = {
        "rollout_num": rollout_num,
        "num_questions": n_q,
        "total_generations": len(detailed),
        "avg_acc": _safe_div(overall_avg, n_q),
        "majority_acc": _safe_div(overall_maj, n_q),
        "best_acc": _safe_div(overall_best, n_q),
        "query_latency_ms_per_gen": (
            f"{(time.time() - t_run) / max(len(detailed), 1) * 1000:.1f}"
        ),
    }

    final = {"overall": overall, "by_task_type": by_task_results}

    # ---- Print ----
    print("\n" + "=" * 80)
    print(f"Overall  ({rollout_num}-rollout, {n_q} questions)")
    print(f"  Avg Acc      : {overall['avg_acc']:.4f}")
    print(f"  Majority Acc : {overall['majority_acc']:.4f}")
    print(f"  Best-of-N Acc: {overall['best_acc']:.4f}")
    print("-" * 80)
    print(f"{'task_type':<22} | {'count':>5} | {'avg':>7} | {'major':>7} | {'best':>7}")
    print("-" * 80)
    for k in TASK_TYPES_ORDER:
        r = by_task_results[k]
        print(
            f"{k:<22} | {r['count']:>5d} | {r['avg_acc']:>7.4f} | "
            f"{r['majority_acc']:>7.4f} | {r['best_acc']:>7.4f}"
        )
    print("=" * 80)

    # ---- save ----
    t = time.localtime()
    tag = f".{args.tag}" if args.tag else ""
    base_name = (
        f"rag_conflicts.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}"
        f".rollout{rollout_num}{tag}"
    )
    output_dir = f"{args.output_dir}/{model_short_name}"
    os.makedirs(output_dir, exist_ok=True)

    detail_path = os.path.join(output_dir, f"{base_name}.results.json")
    metric_path = os.path.join(output_dir, f"{base_name}.metrics.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"Saved:")
    print(f"  details: {detail_path}")
    print(f"  metrics: {metric_path}")


if __name__ == "__main__":
    main()
