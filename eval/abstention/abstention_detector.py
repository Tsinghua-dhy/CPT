#!/usr/bin/env python3
"""
AbstentionBench abstention detector.

Uses LLM-as-a-Judge to determine whether a model has abstained, with two
detection paths:
  1. Fast keyword-based detection.
  2. Deep LLM-based detection.
"""
import os
import re
import time
import httpx
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from prompts import (
    ABSTENTION_KEYWORDS,
    get_abstention_judge_prompt,
    get_correctness_judge_prompt,
    contains_abstention_keyword,
)

# ------------------ OpenAI client (same config as math_equal.py) ------------------
api_4_key = os.environ.get("OPENAI_API_KEY", "")

client_4 = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=api_4_key,
    http_client=httpx.Client(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        follow_redirects=True,
    ),
)


# ------------------ GPT request helpers ------------------
def gpt_request(prompt: str, max_retries: int = 3) -> str:
    """Single GPT request."""
    for attempt in range(max_retries):
        try:
            rst = client_4.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                temperature=0.0,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            return rst.choices[0].message.content.strip()
        except Exception as e:
            print(f"GPT Request Error (attempt {attempt+1}): {e}")
            time.sleep(1)
    return ""


def gpt_batch_request(prompts: List[str], max_threads: int = 256) -> List[str]:
    """Batched GPT request."""
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_idx = {
            executor.submit(gpt_request, prompt): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Task {idx} failed: {e}")
                results[idx] = ""
    return results


# ------------------ Abstention detection core ------------------
def detect_abstention_keyword(response: str) -> Tuple[bool, str]:
    """
    Keyword-based abstention detection.

    Returns:
        (is_abstention, matched_keyword)
    """
    response_lower = response.lower()
    for keyword in ABSTENTION_KEYWORDS:
        if keyword in response_lower:
            return True, keyword
    return False, ""


def detect_abstention_llm(
    question: str,
    response: str,
    ref_answer: str = "[]",
    abstention_label: str = "Unknown",
    use_coconot_style: bool = True
) -> Tuple[bool, str]:
    """
    LLM-based abstention detection (uses the official AbstentionBench prompt).

    Args:
        question: the question.
        response: the model's reply.
        ref_answer: reference answer.
        abstention_label: ground-truth abstention label.
        use_coconot_style: whether to use the detailed CoCoNot-style prompt.

    Returns:
        (is_abstention, raw_response)
    """
    prompt = get_abstention_judge_prompt(
        question=question,
        model_answer=response,
        ref_answer=ref_answer,
        abstention_label=abstention_label,
        use_coconot_style=use_coconot_style,
    )
    result = gpt_request(prompt)

    # Parse the result – AbstentionBench's official prompt returns "Yes" or "No".
    result_lower = result.strip().lower()

    if result_lower.startswith("yes"):
        return True, result
    elif result_lower.startswith("no"):
        return False, result
    else:
        # Fall back to keyword detection if the judge output is unclear.
        is_abstention, _ = detect_abstention_keyword(response)
        return is_abstention, f"LLM judge unclear ('{result}'), fallback to keyword"


def detect_abstention_hybrid(
    question: str,
    response: str,
    use_llm: bool = True,
    ref_answer: str = "[]",
    abstention_label: str = "Unknown",
    use_coconot_style: bool = True
) -> Dict:
    """
    Hybrid abstention detection: keyword first, then LLM.

    Args:
        question: the question.
        response: the model's reply.
        use_llm: whether to invoke the LLM judge.
        ref_answer: reference answer.
        abstention_label: ground-truth abstention label.
        use_coconot_style: whether to use the detailed CoCoNot-style prompt.

    Returns:
        {
            "is_abstention": bool,
            "method": "keyword" or "llm",
            "reasoning": str,
            "keyword_match": str or None
        }
    """
    # Try keyword detection first.
    keyword_abstention, matched_keyword = detect_abstention_keyword(response)

    if keyword_abstention:
        return {
            "is_abstention": True,
            "method": "keyword",
            "reasoning": f"Matched abstention keyword: '{matched_keyword}'",
            "keyword_match": matched_keyword,
        }

    # Keyword detection found nothing -> fall back to the LLM judge if enabled.
    if use_llm:
        llm_abstention, reasoning = detect_abstention_llm(
            question, response, ref_answer, abstention_label, use_coconot_style
        )
        return {
            "is_abstention": llm_abstention,
            "method": "llm",
            "reasoning": reasoning,
            "keyword_match": None,
        }

    return {
        "is_abstention": False,
        "method": "keyword",
        "reasoning": "No abstention keyword found",
        "keyword_match": None,
    }


def batch_detect_abstention(
    data_items: List[Dict],
    use_llm: bool = True,
    question_key: str = "question",
    response_key: str = "response",
    ref_answer_key: str = "reference_answers",
    abstention_label_key: str = "should_abstain",
    use_coconot_style: bool = True,
    max_threads: int = 256,
) -> List[Dict]:
    """
    Batched abstention detection (uses the official AbstentionBench prompt).

    Args:
        data_items: list of items, each containing question / response.
        use_llm: whether to also run the LLM judge.
        question_key: key for the question text.
        response_key: key for the model response.
        ref_answer_key: key for the reference answer.
        abstention_label_key: key for the ground-truth abstention label.
        use_coconot_style: whether to use the detailed CoCoNot-style prompt.

    Returns:
        list of detection results.
    """
    results = []

    # Pass 1: keyword detection.
    items_need_llm = []
    items_need_llm_indices = []

    for i, item in enumerate(data_items):
        question = item.get(question_key, "")
        response = item.get(response_key, "")

        keyword_abstention, matched_keyword = detect_abstention_keyword(response)

        if keyword_abstention:
            results.append({
                "is_abstention": True,
                "method": "keyword",
                "reasoning": f"Matched: '{matched_keyword}'",
                "keyword_match": matched_keyword,
            })
        else:
            results.append(None)  # placeholder, filled in pass 2
            if use_llm:
                items_need_llm.append(item)
                items_need_llm_indices.append(i)

    # Pass 2: batched LLM judge.
    if use_llm and items_need_llm:
        print(f"Using LLM to detect abstention for {len(items_need_llm)} items...")

        prompts = []
        for item in items_need_llm:
            # Pull the reference answer + ground-truth abstention label.
            ref_answer = item.get(ref_answer_key, "[]")
            if isinstance(ref_answer, list):
                ref_answer = str(ref_answer)

            abstention_label = item.get(abstention_label_key, "Unknown")
            if isinstance(abstention_label, bool):
                abstention_label = str(abstention_label)

            prompt = get_abstention_judge_prompt(
                question=item.get(question_key, ""),
                model_answer=item.get(response_key, ""),
                ref_answer=ref_answer,
                abstention_label=abstention_label,
                use_coconot_style=use_coconot_style,
            )
            prompts.append(prompt)

        llm_results = gpt_batch_request(prompts, max_threads=max_threads)

        for idx, llm_result in zip(items_need_llm_indices, llm_results):
            result_lower = llm_result.strip().lower()

            # AbstentionBench's official prompt returns "Yes" or "No".
            if result_lower.startswith("yes"):
                is_abstention = True
            else:
                is_abstention = False

            results[idx] = {
                "is_abstention": is_abstention,
                "method": "llm",
                "reasoning": llm_result,
                "keyword_match": None,
            }

    # Fill placeholders for items where the LLM judge was skipped.
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "is_abstention": False,
                "method": "keyword",
                "reasoning": "No abstention detected",
                "keyword_match": None,
            }

    return results


# ------------------ Correctness detection ------------------
def detect_correctness_llm(
    question: str,
    reference_answers: List[str],
    prediction: str,
    is_math: bool = False
) -> Tuple[bool, str]:
    """
    LLM-based correctness detection (uses the official AbstentionBench prompt).

    Args:
        question: the question.
        reference_answers: list of acceptable reference answers.
        prediction: the model's prediction.
        is_math: whether this is a math item.

    Returns:
        (is_correct, raw_response)
    """
    if reference_answers is None or (isinstance(reference_answers, list) and len(reference_answers) == 0):
        return False, "No reference answer (should abstain)"

    prompt = get_correctness_judge_prompt(question, reference_answers, prediction, is_math=is_math)
    result = gpt_request(prompt)

    result_lower = result.strip().lower()

    # AbstentionBench's official prompt returns "correct" or "incorrect".
    if result_lower.startswith("correct"):
        return True, result
    else:
        return False, result


def batch_detect_correctness(
    data_items: List[Dict],
    question_key: str = "question",
    reference_key: str = "reference_answers",
    prediction_key: str = "prediction",
    is_math: bool = False,
    max_threads: int = 256,
) -> List[Dict]:
    """
    Batched correctness detection.

    Args:
        data_items: list of items.
        question_key: key for the question text.
        reference_key: key for the reference answer.
        prediction_key: key for the predicted answer.
        is_math: whether these are math items.
    """
    prompts = []
    valid_indices = []

    for i, item in enumerate(data_items):
        ref_answers = item.get(reference_key)
        if ref_answers is None or (isinstance(ref_answers, list) and len(ref_answers) == 0):
            continue

        prompt = get_correctness_judge_prompt(
            item.get(question_key, ""),
            ref_answers,
            item.get(prediction_key, ""),
            is_math=is_math,
        )
        prompts.append(prompt)
        valid_indices.append(i)

    # Dispatch concurrently.
    results = [None] * len(data_items)

    if prompts:
        llm_results = gpt_batch_request(prompts, max_threads=max_threads)

        for idx, llm_result in zip(valid_indices, llm_results):
            result_lower = llm_result.strip().lower()

            # AbstentionBench's official prompt returns "correct" or "incorrect".
            is_correct = result_lower.startswith("correct")

            results[idx] = {
                "is_correct": is_correct,
                "reasoning": llm_result,
                "raw_response": llm_result,
            }

    # Fill items that had no reference answer.
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "is_correct": False,
                "reasoning": "No reference answer (should abstain)",
                "raw_response": "",
            }

    return results


# ------------------ AbstentionBench metrics ------------------
def calculate_abstention_metrics(
    predictions: List[Dict],
    gold_labels: List[bool]
) -> Dict:
    """
    Compute AbstentionBench evaluation metrics.

    Args:
        predictions: list of detection results, each with an `is_abstention` field.
        gold_labels: list of `should_abstain` ground-truth labels.

    Returns:
        a dict containing precision / recall / F1 / accuracy / abstention rate.
    """
    assert len(predictions) == len(gold_labels)

    n = len(predictions)

    # Confusion matrix.
    tp = 0  # True Positive:  should abstain AND model abstained
    fp = 0  # False Positive: should NOT abstain BUT model abstained
    tn = 0  # True Negative:  should NOT abstain AND model did not abstain
    fn = 0  # False Negative: should abstain BUT model did not abstain

    for pred, gold in zip(predictions, gold_labels):
        pred_abstain = pred.get("is_abstention", False)

        if gold and pred_abstain:
            tp += 1
        elif not gold and pred_abstain:
            fp += 1
        elif not gold and not pred_abstain:
            tn += 1
        else:  # gold and not pred_abstain
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    accuracy = (tp + tn) / n if n > 0 else 0
    abstention_rate = (tp + fp) / n if n > 0 else 0

    return {
        "total": n,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "abstention_rate": abstention_rate,
        "should_abstain_count": tp + fn,
        "should_not_abstain_count": tn + fp,
    }


# ------------------ Smoke test ------------------
if __name__ == "__main__":
    test_cases = [
        {
            "question": "What is 2+2?",
            "response": "4"
        },
        {
            "question": "Who will win the 2030 World Cup?",
            "response": "I don't know. This is a future event that hasn't happened yet."
        },
        {
            "question": "Why do fish live on trees?",
            "response": "This question contains a false premise. Fish do not live on trees; they live in water."
        },
        {
            "question": "Which is better?",
            "response": "Could you please clarify what you're comparing? I need more information to answer this question."
        }
    ]

    print("Testing abstention detection...")
    print("=" * 80)

    for i, case in enumerate(test_cases):
        result = detect_abstention_hybrid(case["question"], case["response"], use_llm=False)
        print(f"\nTest {i+1}:")
        print(f"Question: {case['question']}")
        print(f"Response: {case['response'][:100]}...")
        print(f"Result: {result}")
