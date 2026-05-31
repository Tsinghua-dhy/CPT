import os
import re
import time
import httpx
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

# ------------------ OpenAI client ------------------
api_4_key = os.environ.get("OPENAI_API_KEY", "")

client_4 = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=api_4_key,
    http_client=httpx.Client(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        follow_redirects=True,
    ),
)

# ------------------ GPT evaluation ------------------
def gpt4omini_request(prompt: str) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rst = client_4.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                temperature=0.0,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return rst.choices[0].message.content.strip()
        except Exception as e:
            print("ChatGPT ERROR:", e)
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return "Judgment: Incorrect"  # default after retries fail
    
    return "Judgment: Incorrect"

def GPT4omini_batch_request(prompts: List[str], max_threads: int = 128) -> List[str]:
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
    return results

def get_prompt(question, gold, pred):
    return f"""You are an expert math evaluator.
Given a question, a gold answer and a predicted answer, judge if they are mathematically consistent.

Ignore formatting (e.g., \\text{{}}, spacing, capitalization).
Accept equivalent expressions (e.g., factored vs expanded form).
Only when the predicted answer is mathematically consistent with the gold answer, output "Correct".

Output format:
Reason: Brief explanation
Judgment: Correct / Incorrect

Input:
Question: {question}
Gold: {gold}
Pred: {pred}"""

# ------------------ math_verify wrapper ------------------
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    # Check whether ground_truth already contains \boxed{}
    if "\\boxed{" in ground_truth:
        ground_truth_boxed = ground_truth
    else:
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass
    return ret_score

# ------------------ Main entry ------------------
def eval_math_with_gpt(
    golden_answers_list: List[str],
    pred_answers_list: List[str],
    questions: Optional[List[str]] = None
) -> List[Dict]:
    assert len(golden_answers_list) == len(pred_answers_list)
    n = len(golden_answers_list)
    results = []

    # Step 1: batched math_verify
    print(f"Running math_verify on {n} samples...")
    for gold, pred in zip(golden_answers_list, pred_answers_list):
        score = compute_score(pred, gold)
        results.append({"is_equal": score, "is_equal_gpt": None})

    # Step 2: collect items that did not pass
    need_gpt_idx = [i for i, r in enumerate(results) if r["is_equal"] == 0]
    if not need_gpt_idx:
        for r in results:
            r["is_equal_gpt"] = 1
        return results

    print(f"{len(need_gpt_idx)} samples failed math_verify, sending to GPT...")

    # Step 3: GPT batchcheck
    prompts = [
        get_prompt(questions[i] if questions else "N/A",
                   golden_answers_list[i],
                   pred_answers_list[i])
        for i in need_gpt_idx
    ]
    gpt_outputs = GPT4omini_batch_request(prompts)

    # Step 4: parse GPT output
    for idx, out in zip(need_gpt_idx, gpt_outputs):
        match = re.search(r"Judgment:\s*(Correct|Partially correct|Incorrect)", out)
        judgment = match.group(1) if match else "Incorrect"
        tag = {"Correct": 1, "Partially correct": 0.5, "Incorrect": 0}.get(judgment, 0)
        results[idx]["is_equal_gpt"] = tag

    # Items with math_verify==1 are scored 1 directly
    for i in range(n):
        if results[i]["is_equal_gpt"] is None:
            results[i]["is_equal_gpt"] = 1

    return results


# ------------------ Example ------------------
#if __name__ == "__main__":
#    golden = ["x=2", "3", "1+2"]
#    pred = ["x=2", "4", "3"]
#    q = ["Solve for x.", "Compute the value.", "Simplify."]
#    rst = eval_math_with_gpt(golden, pred, q)
#    print(json.dumps(rst, indent=2, ensure_ascii=False))
