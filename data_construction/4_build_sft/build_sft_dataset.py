#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the math-path-judge SFT training data (v2).
Extract every highest-confidence judgment as a model response (instead of picking one at random). 
"""

import argparse
import json
import os
import random
import pandas as pd
from transformers import AutoTokenizer
from collections import Counter


def build_prompt(question, ground_truth, answer_a, reasoning_a, answer_b, reasoning_b):
    """Build the math-path-judge prompt."""
    
    INSTRUCTION_TMPL = """You are an expert mathematical reasoning evaluator. You will be given:
1. A mathematical problem
2. The ground truth (correct answer)
3. Two reasoning paths (Path A and Path B) with their predicted answers

Your task is to determine which reasoning path is better by considering:
- **Correctness**: Does the path lead to the correct answer?
- **Logical Soundness**: Is the reasoning logically valid and free of mathematical errors?
- **Clarity**: Is the explanation clear and easy to follow?
- **Efficiency**: Does it solve the problem without unnecessary steps?

You must choose one of the following options:
- Path A is better
- Path B is better
- Both are equally good
- Both are equally bad

**Input:**

Question:
{question}

Ground Truth Answer:
{ground_truth}

Path A:
Predicted Answer: {answer_a}
Reasoning:
{reasoning_a}

Path B:
Predicted Answer: {answer_b}
Reasoning:
{reasoning_b}"""
    
    # Format the instruction body
    body = INSTRUCTION_TMPL.format(
        question=question,
        ground_truth=ground_truth,
        answer_a=answer_a,
        reasoning_a=reasoning_a,
        answer_b=answer_b,
        reasoning_b=reasoning_b
    )
    
    # Use the standard User/Assistant scaffold
    prompt = (
        "The User asks a question, and the Assistant solves it.\n"
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.\n"
        "The final answer should be enclosed within \\boxed{}.\n"
        "User: " + body + "\n"
        "Assistant:"
    )
    
    return prompt


def get_all_highest_confidence_answers(judge_results):
    """
    Return every highest-confidence judgment together with its analysis.
    
    return: list of (judgment, analysis) or emptylist
    """
    all_judgments = judge_results.get("all_judgments", [])
    all_analyses = judge_results.get("all_analyses", [])
    all_confidences = judge_results.get("all_confidences", [])
    
    if not all_judgments or not all_analyses or not all_confidences:
        return []
    
    # Confidence -> score map
    confidence_map = {
        "Very High": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1
    }
    
    # Find the highest-confidence value among non-empty judgments
    valid_data = []
    
    for judgment, analysis, confidence in zip(all_judgments, all_analyses, all_confidences):
        # Skip empty judgments
        if judgment and str(judgment).strip():
            confidence_score = confidence_map.get(confidence, 0)
            valid_data.append((judgment, analysis, confidence_score))
    
    # If no valid judgments exist, return an empty list
    if not valid_data:
        return []
    
    # Find the maximum confidence value
    max_confidence = max(item[2] for item in valid_data)
    
    # Return every (judgment, analysis) pair with that maximum confidence
    return [(item[0], item[1]) for item in valid_data if item[2] == max_confidence]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", 
                        default="./post_training/sft_data.jsonl",
                        help="training-set input JSONL file")
    parser.add_argument("--local_dir", 
                        default="./output",
                        help="output directory")
    parser.add_argument("--tokenizer_path", 
                        default="Qwen/Qwen3-14B-Base",
                        help="Tokenizerpath")
    parser.add_argument("--max_length", 
                        type=int, 
                        default=12288,
                        help="max token length")
    parser.add_argument("--random_seed",
                        type=int,
                        default=42,
                        help="random seed")
    args = parser.parse_args()

    # Set the random seed (used to randomize the answer formatting)
    random.seed(args.random_seed)

    print("=" * 60)
    print("Build the math-path-judge SFT training data (v2).")
    print("Extracting every highest-confidence judgment")
    print("=" * 60)
    
    # loadtokenizer
    print(f"\nloadtokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Make sure the output directory exists
    os.makedirs(args.local_dir, exist_ok=True)
    
    LENGTH_THRESHOLD = args.max_length
    
    # Process the training set
    print(f"\nProcessing training set: {args.train_file}")
    train_data_list = []
    with open(args.train_file, "r", encoding="utf-8") as f:
        for line in f:
            train_data_list.append(json.loads(line.strip()))
    
    print(f"Read {len(train_data_list)} raw items")
    
    train_data_all = []
    train_skipped_no_judgment = 0
    train_skipped_too_long = 0
    total_max_confidence_samples = 0
    
    for idx, example in enumerate(train_data_list):
        if idx % 1000 == 0:
            print(f"  progress: {idx}/{len(train_data_list)}")
        
        # extractfield
        question = example.get("question", "")
        ground_truth = example.get("ground_truth", "")
        
        path_a = example.get("path_a", {})
        path_b = example.get("path_b", {})
        
        answer_a = path_a.get("pred_ans", "")
        reasoning_a = path_a.get("gen_text", "")
        answer_b = path_b.get("pred_ans", "")
        reasoning_b = path_b.get("gen_text", "")
        
        # Get the judgment results
        judge_results = example.get("judge_results", {})
        
        # Get every highest-confidence judgment and analysis
        highest_confidence_pairs = get_all_highest_confidence_answers(judge_results)
        
        if not highest_confidence_pairs:
            train_skipped_no_judgment += 1
            continue
        
        total_max_confidence_samples += len(highest_confidence_pairs)
        
        # Build the prompt (the prompt is identical for the same question)
        prompt = build_prompt(question, ground_truth, answer_a, reasoning_a, answer_b, reasoning_b)
        
        # Emit one training sample per highest-confidence judgment
        for judgment, analysis in highest_confidence_pairs:
            # Build the response (analysis + final answer)
            rand_val = random.random()
            if rand_val < 1/3:
                ans_part = f"\n\n**Final Answer:** \\boxed{{{judgment}}}" 
            elif rand_val < 2/3:
                ans_part = f" The final answer is \\boxed{{{judgment}}}"
            else:
                ans_part = f"\n\n\\boxed{{{judgment}}}"
            response = analysis + ans_part
            
            # Check the length
            if len(tokenizer.encode(prompt + response, add_special_tokens=False)) > LENGTH_THRESHOLD:
                train_skipped_too_long += 1
                continue
            
            data = {
                "data_source": "math_path_judge_sft_v2",
                "prompt": prompt,
                "response": response,
                "ability": "math_reasoning_evaluation",
                "judgment": judgment,
                "difficulty": example.get("difficulty", "unknown"),
                "pair_type": example.get("pair_type", "unknown"),
                "task_type": example.get("task_type", "unknown")
            }
            train_data_all.append(data)
    
    print(f"\nTraining-set processing done:")
    print(f"  valid training samples: {len(train_data_all)}")
    print(f"  raw items: {len(train_data_list)}")
    print(f"  extracted highest-confidence samples: {total_max_confidence_samples}")
    print(f"  skipped (no valid judgment): {train_skipped_no_judgment}")
    print(f"  skipped (too long): {train_skipped_too_long}")
    print(f"  expansion ratio: {len(train_data_all)/len(train_data_list):.2f}x")
    
    # Save the training set
    train_jsonl_path = os.path.join(args.local_dir, "train.jsonl")
    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for item in train_data_all:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    train_df = pd.DataFrame(train_data_all)
    train_df.to_parquet(os.path.join(args.local_dir, "train.parquet"), index=False)
    print(f"\nTraining set saved to: {args.local_dir}/train.{{jsonl,parquet}}")
    
    # Print stats
    print("\n" + "=" * 60)
    print("Data statistics")
    print("=" * 60)
    print(f"training set: {len(train_data_all)} items")
    
    # Judgment-result distribution
    if train_data_all:
        print("\nJudgment-result distribution:")
        train_judgments = [item["judgment"] for item in train_data_all]
        judgment_counter = Counter(train_judgments)
        for judgment, count in judgment_counter.most_common():
            print(f"  {judgment}: {count} ({count/len(train_data_all)*100:.2f}%)")
        
        print("\nDifficulty distribution:")
        difficulties = [item["difficulty"] for item in train_data_all]
        difficulty_counter = Counter(difficulties)
        for difficulty, count in difficulty_counter.most_common():
            print(f"  {difficulty}: {count} ({count/len(train_data_all)*100:.2f}%)")
    
    print("=" * 60)
    print("\ndone! ")
