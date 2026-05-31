import os
"""
update from https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py and https://github.com/mandarjoshi90/triviaqa/blob/master/triviaqa_evaluation.py
"""

import pdb
import sys
try:
    import ujson as json
except ImportError:
    import json
import re
import string
from collections import Counter
import pickle
import jsonlines
import time
import httpx
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )


def cover_exact_match_score_1(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")
    # print("prediction: ",prediction)
    # print("ground_truth: ",ground_truth)
    # print("pre_list: ",pre_list)
    # print("ground_list: ",ground_list)
    # Order- and contiguity-insensitive
    return all(ground in pre_list for ground in ground_list)


def cover_exact_match_score_2(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")

    for i in range(len(pre_list) - len(ground_list) + 1):
        if pre_list[i : i + len(ground_list)] == ground_list:
            return True
    pre_str = " ".join(pre_list)
    ground_str = " ".join(ground_list)
    if ground_str in pre_str:
        return True
    return False


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    if metric_fn.__name__ == "exact_match_score":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "f1_score":
        for ground_truth in ground_truths:
            f1, prec, recall = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append((f1, prec, recall))
        f1, prec, recall = max(scores_for_ground_truths, key=lambda x: x[0])
        return f1, prec, recall
    elif metric_fn.__name__ == "cover_exact_match_score_1":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "cover_exact_match_score_2":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


# LLM-as-Judge helpers

#your openai api here for llm-as-a-judge
api_4_key = os.environ.get("OPENAI_API_KEY", "")

# create a synchronous OpenAI client
client_4 = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=api_4_key,
    http_client=httpx.Client(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        follow_redirects=True,
    ),
)

def get_judge_prompt(question, gold_answer, predicted_answer):
    """Build the LLM-judge prompt (with CoT and few-shot demos)."""
    return f"""You are an expert answer evaluator.
Given a question, a gold answer and a predicted answer, judge if the predicted answer is correct and consistent with the gold answer.

Guidelines:
- Ignore minor formatting differences (e.g., punctuation, spacing, capitalization).
- Accept semantically equivalent expressions (e.g., "No" vs "No, it is not endangered").
- Focus on whether the key information and meaning align.
- The predicted answer is correct if it captures the essential information from the gold answer.

Output format:
Reason: Brief explanation of your judgment
Judgment: Correct / Incorrect

Examples:

Input:
Question: When does this year's Passover start?
Gold: begins at sundown on Saturday, April 12.
Pred: Saturday, April 12

Output:
Reason: The predicted answer captures the key date information (Saturday, April 12) from the gold answer. The missing detail about "sundown" is minor and doesn't change the core answer.
Judgment: Correct

Input:
Question: Is the Giant Panda still considered an endangered species?
Gold: No, the Giant Panda is no longer considered an endangered species. It was reclassified as "vulnerable" in 2016 by the IUCN.
Pred: No

Output:
Reason: The predicted answer "No" directly answers the question correctly. While it lacks the additional context about rebucket, the core answer is accurate and complete.
Judgment: Correct

Now evaluate:

Input:
Question: {question}
Gold: {gold_answer}
Pred: {predicted_answer}

Output:"""

def gpt4omini_request(prompt):
    """Single GPT request."""
    while True:
        try:
            rst = client_4.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                temperature=0.0,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}]
            )
            return rst.choices[0].message.content.strip()
        except Exception as e:
            print("ChatGPT ERROR:", e)
            time.sleep(1)

def GPT4omini_batch_request(prompts, max_threads=128):
    """batchGPTrequest"""
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_idx = {
            executor.submit(gpt4omini_request, prompt): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Task {idx} failed:", e)
                results[idx] = "Incorrect"  # default to "Incorrect"
    return results

def llm_as_judge_score(question, prediction, ground_truth):
    """Use the LLM as a judge for scoring."""
    prompt = get_judge_prompt(question, ground_truth, prediction)
    result = gpt4omini_request(prompt)
    # Parse the result, return a bool
    # Supports several formats: Correct/Incorrect, True/False
    result_lower = result.lower().strip()
    
    # If the response contains "incorrect" or "false", return False
    if "incorrect" in result_lower or result_lower == "false":
        return False
    # Check for "judgment: correct" or a standalone "correct"
    if "judgment: correct" in result_lower or result_lower == "correct" or result_lower == "true":
        return True
    
    # Default to False (conservative)
    return False

def update_answer_with_llm_judge(metrics, question, prediction, gold, use_llm_judge=False):
    """Update the answer evaluation result, including the LLM-judge option."""
    em = metric_max_over_ground_truths(exact_match_score, prediction, gold)
    f1, prec, recall = metric_max_over_ground_truths(f1_score, prediction, gold)
    cover_em_1 = metric_max_over_ground_truths(
        cover_exact_match_score_1, prediction, gold
    )
    cover_em_2 = metric_max_over_ground_truths(
        cover_exact_match_score_2, prediction, gold
    )

    # LLM-as-Judge scoring - only when F1 != 1.0
    llm_judge_score_val = 0
    if use_llm_judge and f1 < 1.0:
        # For multiple gold answers, take the max score
        llm_scores = []
        for gt in gold:
            score = llm_as_judge_score(question, prediction, gt)
            llm_scores.append(score)
        llm_judge_score_val = max(llm_scores) if llm_scores else 0
    elif use_llm_judge and f1 == 1.0:
        # F1 == 1.0 -> score 1 directly
        llm_judge_score_val = 1

    metrics["em"] += float(em)
    metrics["cover_em_1"] += float(cover_em_1)
    metrics["cover_em_2"] += float(cover_em_2)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall
    if use_llm_judge:
        metrics["llm_judge"] += float(llm_judge_score_val)

    if cover_em_1:
        metrics["acc_num"] += 1
    
    return em, prec, recall, f1, cover_em_1, cover_em_2, llm_judge_score_val

def batch_llm_as_judge_evaluation(data_items):
    """Batched LLM evaluation - only items with F1 < 1.0.
    
    Returns:
        list of dict: each element contains {"score": bool, "raw_response": str, "reason": str}
    """
    # First compute F1 for every sample; pick those needing LLM evaluation
    items_to_eval = []
    item_indices = []
    
    for i, item in enumerate(data_items):
        prediction = item["pred_ans"]
        if isinstance(item["answer"], list):
            gold_answers = item["answer"]
        else:
            gold_answers = [item["answer"]]
        
        # Compute F1
        f1, _, _ = metric_max_over_ground_truths(f1_score, prediction, gold_answers)
        
        if f1 < 1.0:  # Only items with F1 < 1.0 need LLM evaluation
            items_to_eval.append(item)
            item_indices.append(i)
    
    print(f"LLM Judge: Evaluating {len(items_to_eval)} out of {len(data_items)} samples (F1 < 1.0)")
    
    if not items_to_eval:
        # If nothing needs evaluation, return all-1 results
        return [{"score": 1.0, "raw_response": "F1=1.0, skipped LLM evaluation", "reason": "Perfect F1 match"}] * len(data_items)
    
    # Build prompts for the samples that need evaluation
    prompts = []
    prompt_metadata = []  # records (item_idx_in_eval, gold_answer_idx)
    
    for eval_idx, item in enumerate(items_to_eval):
        question = item["question"]
        prediction = item["pred_ans"]
        
        # Handle multi-gold cases
        if isinstance(item["answer"], list):
            gold_answers = item["answer"]
        else:
            gold_answers = [item["answer"]]
        
        # Build a prompt for each gold answer
        for gold_idx, gold_answer in enumerate(gold_answers):
            prompt = get_judge_prompt(question, gold_answer, prediction)
            prompts.append(prompt)
            prompt_metadata.append((eval_idx, gold_idx))
    
    # batchrequest
    results = GPT4omini_batch_request(prompts)
    
    # Organize the results by eval_idx
    llm_results = {}  # {eval_idx: [(raw_response, is_correct, reason), ...]}
    for i, result in enumerate(results):
        eval_idx, gold_idx = prompt_metadata[i]
        if eval_idx not in llm_results:
            llm_results[eval_idx] = []
        
        # parseresult
        result_lower = result.lower().strip()
        
        # Correct check logic: scan for 'incorrect' first, then 'correct'
        if "incorrect" in result_lower or result_lower == "false":
            is_correct = False
        elif "judgment: correct" in result_lower or result_lower == "correct" or result_lower == "true":
            is_correct = True
        else:
            # Default to False (conservative)
            is_correct = False
        
        # extract Reason (ifhas ) 
        reason = ""
        if "reason:" in result_lower:
            try:
                reason_part = result.split("Reason:", 1)[1].split("Judgment:", 1)[0].strip()
                reason = reason_part
            except:
                reason = result
        else:
            reason = result
        
        llm_results[eval_idx].append({
            "raw_response": result,
            "is_correct": is_correct,
            "reason": reason
        })
    
    # Build the complete result array
    final_results = []
    eval_idx = 0
    
    for i in range(len(data_items)):
        if i in item_indices:  # This sample needed LLM evaluation
            if eval_idx in llm_results:
                # Take the max across all gold answers
                all_responses = llm_results[eval_idx]
                best_response = max(all_responses, key=lambda x: x["is_correct"])
                final_results.append({
                    "score": float(best_response["is_correct"]),
                    "raw_response": best_response["raw_response"],
                    "reason": best_response["reason"],
                    "all_responses": [r["raw_response"] for r in all_responses] if len(all_responses) > 1 else None
                })
            else:
                final_results.append({
                    "score": 0.0,
                    "raw_response": "LLM evaluation failed",
                    "reason": "Error during evaluation"
                })
            eval_idx += 1
        else:  # F1 == 1.0 -> set True directly
            final_results.append({
                "score": 1.0,
                "raw_response": "F1=1.0, skipped LLM evaluation",
                "reason": "Perfect F1 match"
            })
    
    return final_results

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, "r") as reader:
        for obj in reader:
            data.append(obj)
    return data

def eval(file, use_llm_judge=True, batch_llm_eval=True):
    data = read_jsonl(file)[:2000]
    print(len(data))
    print(f"Eval {len(data)} from {file}")
    
    metrics = {
        "em": 0,
        "f1": 0,
        "cover_em_1": 0,
        "cover_em_2": 0,
        "prec": 0,
        "recall": 0,
        "acc_num": 0
    }
    
    if use_llm_judge:
        metrics["llm_judge"] = 0
    
    # ifusebatchLLMevaluation
    if use_llm_judge and batch_llm_eval:
        print("Using batch LLM evaluation...")
        llm_judge_results = batch_llm_as_judge_evaluation(data)
    
    for i, d in enumerate(data):
        pred_answer = d["pred_ans"]
        question = d.get("question", "")

        if isinstance(d["answer"], list):
            gold_answers = d["answer"]
        else:
            gold_answers = [d["answer"]]
        
        if use_llm_judge and not batch_llm_eval:
            # Single-evaluation mode
            em, prec, recall, f1, cover_em_1, cover_em_2, llm_judge_score_val = update_answer_with_llm_judge(
                metrics, question, pred_answer, gold_answers, use_llm_judge=True
            )
        elif use_llm_judge and batch_llm_eval:
            # Batched-evaluation mode, use the pre-computed results
            em, prec, recall, f1, cover_em_1, cover_em_2, _ = update_answer_with_llm_judge(
                metrics, question, pred_answer, gold_answers, use_llm_judge=False
            )
            llm_judge_score_val = llm_judge_results[i]["score"]
            metrics["llm_judge"] += float(llm_judge_score_val)
            
            # Add debug info
            if f1 < 1.0 and llm_judge_score_val:
                print(f"Sample {i}: F1={f1:.3f}, LLM-Judge=True - Rescued by LLM!")
        else:
            # Skip LLM evaluation
            em, prec, recall, f1, cover_em_1, cover_em_2, llm_judge_score_val = update_answer_with_llm_judge(
                metrics, question, pred_answer, gold_answers, use_llm_judge=False
            )
            llm_judge_score_val = 0

        # Optional: print mismatched cases (for debugging)
        # if not cover_em_2:
        #     if d.get("gpt4o_output") == "True":
        #         print("==="*40)
        #         print("ques:", d["question"])
        #         print("pred:", pred_answer)
        #         print("gold:", d["answer"])
        #         print(f"f1:{f1}, cover_em_1:{cover_em_1}, llm_judge:{llm_judge_score_val}")
        #         print("==="*40)

    N = len(data)
    for k in metrics.keys():
        if k == "acc_num":
            continue
        metrics[k] /= N

    # Prepare the final result
    final_metrics = [
        str(round(metrics['em']*100, 1)), 
        str(round(metrics["cover_em_1"]*100, 1)), 
        str(round(metrics['f1']*100, 1)),
        str(metrics["acc_num"]),
        str(round(metrics["cover_em_2"]*100, 1))
    ]
    
    if use_llm_judge:
        final_metrics.append(str(round(metrics["llm_judge"]*100, 1)))

    print("Eval File: ", file)
    print("EM: ", final_metrics[0])
    print("Cover-EM: ", final_metrics[1])
    print("Cover-EM_2: ", final_metrics[4])
    print("F1: ", final_metrics[2])
    print("Acc_Num: ", final_metrics[3])
    if use_llm_judge:
        print("LLM-Judge: ", final_metrics[5])

    overall_results = {
        'EM': final_metrics[0],
        'Cover-EM': final_metrics[1],
        'Cover-EM_2': final_metrics[4],
        'F1': final_metrics[2],
        'Acc_Num': final_metrics[3],
    }
    
    if use_llm_judge:
        overall_results['LLM-Judge'] = final_metrics[5]

    final_metrics_dict = {'overall': overall_results}
    output_file = file.replace(".jsonl", ".metrics.json")
    with open(output_file, mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics_dict, json_file, indent=4, ensure_ascii=False)
    
    return overall_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metric_eval.py <file_path> [--llm-judge] [--batch-llm]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    use_llm_judge = "--llm-judge" in sys.argv
    batch_llm_eval = "--batch-llm" in sys.argv
    
    result = eval(file_path, use_llm_judge=use_llm_judge, batch_llm_eval=batch_llm_eval)
    print("Final Results:", result)

# Usage examples:
# python metric_eval.py /path/to/file.jsonl                # without LLM judge
# python metric_eval.py /path/to/file.jsonl --llm-judge   # with LLM judge (single-eval mode)
# python metric_eval.py /path/to/file.jsonl --llm-judge --batch-llm  # useLLM judge (batchevaluation)