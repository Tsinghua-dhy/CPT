import argparse
from collections import defaultdict
import copy
import json
import json
import multiprocessing
import os
import random
import re
import sys
import time
from time import sleep

import requests
from tqdm import tqdm

from datasets import Dataset
# Make eval/utils/ importable so we can reuse `math_equal.py`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from math_equal import eval_math_with_gpt, compute_score
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--src_files", type=str, nargs='+', required=True)
    parser.add_argument("--rollout_nums", type=int, nargs='+', default=None,
                        help="Rollout times for each src_file, e.g., --rollout_nums 8 4 1")
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3")
    parser.add_argument("--model_path", type=str, default="None")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--port", type=str, default="5004")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--use_llm_judge", action="store_true",
                        help="Use LLM as a judge for evaluation (GPT-based). Only effective for math500, minervamath, olympiad datasets. Other datasets always use rule-based evaluation.")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Force using chat template for instruct models. If not set, auto-detect based on model name.")
    return parser.parse_args()


def is_instruct_model(model_name):
    """Auto-detect if the model is an instruct model based on model name."""
    model_name_lower = model_name.lower()
    instruct_keywords = ['instruct', 'distill', 'qwen2.5', 'chat', 'it']
    for keyword in instruct_keywords:
        if keyword in model_name_lower:
            return True
    return False


def process_text(example, model_short_name, tokenizer, use_chat_template=False):
    question_raw = example["question"]

    if use_chat_template:
        # Instruct model: use chat template
        instruct_prompt = f"""You are a helpful math problem solving expert. Please solve this problem step by step and put your final answer within \\boxed{{}}.

Question: {question_raw}"""
        prompt_chat = [{"role": "user", "content": instruct_prompt}]
        prompt = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    else:
        # Base model: use raw text directly
        prompt = f"""The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The final answer should be enclosed within \\boxed{{}}.

User:{question_raw}
Assistant:"""

    example["chat_prompt"] = prompt
    return example


def extract_answer_math(s):
    extracted_text = ''

    def extract_boxed_content(text):
        pattern = r'\\boxed\{'
        matches = []
        for match in re.finditer(pattern, text):
            start_pos = match.end() - 1
            brace_count = 0
            content_start = start_pos + 1
            for i in range(start_pos, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        matches.append(text[content_start:i])
                        break
        return matches

    all_matches = []
    boxed_matches = extract_boxed_content(s)
    all_matches.extend(boxed_matches)
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, s, re.DOTALL | re.IGNORECASE)
    all_matches.extend(answer_matches)
    choice_pattern = r'[Tt]he correct answer is:?\s*([A-Ja-j])'
    choice_matches = re.findall(choice_pattern, s, re.IGNORECASE)
    all_matches.extend(choice_matches)
    if all_matches:
        extracted_text = all_matches[-1]
    return extracted_text.strip()


def main():
    print("=Begin="*10)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    t_start = time.time()
    model_path = args.model_path
    model_short_name = model_path.split('/')[-2].lower() + model_path.split('/')[-1].lower()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_gpus = torch.cuda.device_count()
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus,
              gpu_memory_utilization=args.gpu_memory_rate, trust_remote_code=True)

    # Determine whether to use chat template
    use_chat_template = args.use_chat_template or is_instruct_model(model_path)
    print(f"Model: {model_path}")
    print(f"Use chat template: {use_chat_template} (manual: {args.use_chat_template}, auto-detect: {is_instruct_model(model_path)})")

    # Handle rollout_nums
    if args.rollout_nums is None:
        rollout_nums = [1] * len(args.src_files)
    else:
        if len(args.rollout_nums) != len(args.src_files):
            raise ValueError(f"rollout_nums length ({len(args.rollout_nums)}) must match src_files length ({len(args.src_files)})")
        rollout_nums = args.rollout_nums

    print(f"Files and their rollout times:")
    for src_file, rollout_num in zip(args.src_files, rollout_nums):
        print(f"  {src_file}: {rollout_num} rollouts")

    for file_idx, src_file in enumerate(args.src_files):
        rollout_num = rollout_nums[file_idx]
        print(f"\n{'='*80}\nProcessing file: {src_file} (Rollout: {rollout_num}x)\n{'='*80}")

        data_ori_all = []
        with open(src_file, "r") as f:
            for i, line in enumerate(f):
                if args.start_sample <= i < args.end_sample:
                    obj_ori = json.loads(line)
                    data_ori_all.append(obj_ori)
                if i >= args.end_sample - 1:
                    break
        print("All Data Length:", len(data_ori_all))

        # Per-question rollout buffer
        # question_rollouts[question_idx] = [{"pred_ans": ..., "gen_text": ..., "is_equal": ...}, ...]
        question_rollouts = defaultdict(list)

        # ==== Batch-generate all rollouts ====
        print(f"\n{'='*60}\nGenerating {rollout_num} rollouts for all questions in batch\n{'='*60}")

        chunk_size = 20000
        chunk_num = (len(data_ori_all) + chunk_size - 1) // chunk_size

        for h in range(chunk_num):
            print(f"== Begin Chunk {h+1}/{chunk_num} ==")
            data_ori = data_ori_all[h * chunk_size:(h + 1) * chunk_size]
            data = [copy.deepcopy(item) for item in data_ori]
            for item in data:
                item = process_text(item, model_short_name, tokenizer, use_chat_template)
                item["gen_text_store"] = ""

            # Repeat each question's prompt rollout_num times
            prompts = []
            prompt_to_question_map = []  # (question_idx, rollout_idx) per prompt
            for i, item in enumerate(data):
                for rollout_idx in range(rollout_num):
                    prompts.append(item["chat_prompt"])
                    prompt_to_question_map.append((h * chunk_size + i, rollout_idx))

            print(f"  Generating {len(prompts)} outputs ({len(data)} questions x {rollout_num} rollouts)")

            sampling_params = SamplingParams(
                temperature=args.temp, top_p=args.top_p, top_k=args.top_k,
                max_tokens=args.max_tokens, stop=["<|im_end|>", "<|endoftext|>"]
            )
            outputs = llm.generate(prompts, sampling_params)

            # Parse outputs and dispatch back to (question, rollout)
            for output_idx, output in enumerate(outputs):
                global_idx, rollout_idx = prompt_to_question_map[output_idx]
                local_idx = global_idx - h * chunk_size
                q = data[local_idx]["question"]
                a = data[local_idx]["answer"]
                gen_text = output.outputs[0].text.strip()
                pred = extract_answer_math(gen_text)
                if pred == "":
                    pred = "I don't know."

                question_rollouts[global_idx].append({
                    "question": q,
                    "answer": a,
                    "pred_ans": pred,
                    "gen_text_store": gen_text,
                })

        print(f"== Generation done, total {len(question_rollouts)} questions with {rollout_num} rollouts each ==")

        # ==== Batch-evaluate all rollouts ====
        print("\n== Evaluating all rollouts in batch ==")

        # Collect all items to be evaluated
        all_golden_answers = []
        all_pred_answers = []
        all_questions = []
        eval_map = []  # (question_idx, rollout_idx) per evaluation result

        for q_idx in sorted(question_rollouts.keys()):
            rollouts = question_rollouts[q_idx]
            for rollout_idx, rollout in enumerate(rollouts):
                all_golden_answers.append(rollout["answer"])
                all_pred_answers.append(rollout["pred_ans"])
                all_questions.append(rollout["question"])
                eval_map.append((q_idx, rollout_idx))

        print(f"  Total evaluations: {len(all_golden_answers)}")

        # Only math500/minervamath/olympiad use the LLM-as-a-judge fallback when
        # --use_llm_judge is set. Other datasets (amc22/23, aime24/25, ...) always
        # use rule-based evaluation.
        llm_judge_datasets = ["math500", "minervamath", "olympiad", "math_all"]
        dataset_supports_llm_judge = any(ds in src_file.lower() for ds in llm_judge_datasets)

        # Decide based on flag + dataset type
        use_llm_for_this_dataset = args.use_llm_judge and dataset_supports_llm_judge

        if use_llm_for_this_dataset:
            print(f"  Using LLM as a judge (GPT-based evaluation) for dataset: {src_file}")
            # Batch evaluation
            eval_results = eval_math_with_gpt(all_golden_answers, all_pred_answers, all_questions)

            # Dispatch results back to each rollout
            for idx, (q_idx, rollout_idx) in enumerate(eval_map):
                question_rollouts[q_idx][rollout_idx]["is_equal"] = eval_results[idx]["is_equal"]
                question_rollouts[q_idx][rollout_idx]["is_equal_gpt"] = eval_results[idx]["is_equal_gpt"]
        else:
            if args.use_llm_judge and not dataset_supports_llm_judge:
                print(f"  Dataset '{src_file}' does not support LLM judge, using rule-based evaluation only")
            else:
                print("  Using rule-based evaluation only (no LLM judge)")
            # Rule-based evaluation only (no GPT)
            # Wrap pred_ans in \boxed{...} because compute_score expects it
            for idx, (q_idx, rollout_idx) in enumerate(eval_map):
                # Run math equivalence check via compute_score
                # Add \boxed{} if pred_ans does not already contain it
                pred_with_boxed = all_pred_answers[idx]
                if "\\boxed{" not in pred_with_boxed:
                    pred_with_boxed = f"\\boxed{{{pred_with_boxed}}}"
                score = compute_score(pred_with_boxed, all_golden_answers[idx])
                question_rollouts[q_idx][rollout_idx]["is_equal"] = score
                question_rollouts[q_idx][rollout_idx]["is_equal_gpt"] = score  # keep consistent

        # ==== Compute several accuracy metrics ====
        print("\n== Computing metrics ==")

        # 1. Majority Voting (Pass@k) -- the standard
        majority_correct = 0
        majority_correct_gpt = 0

        # 2. Average Accuracy -- mean over rollouts
        all_correct = []
        all_correct_gpt = []

        # 3. Best-of-N -- correct if any rollout is correct
        best_correct = 0
        best_correct_gpt = 0

        # 4. Detailed results to save
        all_finished = []

        for q_idx in sorted(question_rollouts.keys()):
            rollouts = question_rollouts[q_idx]

            # Aggregate this question's rollouts
            is_equal_list = [r["is_equal"] for r in rollouts]
            is_equal_gpt_list = [r["is_equal_gpt"] for r in rollouts]

            # Majority Voting
            majority_vote = 1 if sum(is_equal_list) > len(is_equal_list) / 2 else 0
            majority_vote_gpt = 1 if sum(is_equal_gpt_list) > len(is_equal_gpt_list) / 2 else 0
            majority_correct += majority_vote
            majority_correct_gpt += majority_vote_gpt

            # Average accuracy
            all_correct.extend(is_equal_list)
            all_correct_gpt.extend(is_equal_gpt_list)

            # Best-of-N
            best_correct += (1 if any(is_equal_list) else 0)
            best_correct_gpt += (1 if any(is_equal_gpt_list) else 0)

            # Save detailed results for every rollout
            for rollout_idx, rollout in enumerate(rollouts):
                all_finished.append({
                    "question_idx": q_idx,
                    "rollout_idx": rollout_idx,
                    "question": rollout["question"],
                    "answer": rollout["answer"],
                    "pred_ans": rollout["pred_ans"],
                    "gen_text_store": rollout["gen_text_store"],
                    "Metrics": {
                        "math_equal": rollout["is_equal"],
                        "math_equal_gpt": rollout["is_equal_gpt"],
                        "majority_vote": majority_vote,
                        "majority_vote_gpt": majority_vote_gpt,
                    }
                })

        num_questions = len(question_rollouts)

        overall_results = {
            "rollout_num": rollout_num,
            "num_questions": num_questions,
            "total_generations": len(all_finished),

            # Primary: Majority Voting
            "majority_voting_acc": float(majority_correct / num_questions) if num_questions > 0 else 0.0,
            "majority_voting_acc_gpt": float(majority_correct_gpt / num_questions) if num_questions > 0 else 0.0,

            # Reference: Average Accuracy
            "average_acc": float(np.mean(all_correct)) if len(all_correct) > 0 else 0.0,
            "average_acc_gpt": float(np.mean(all_correct_gpt)) if len(all_correct_gpt) > 0 else 0.0,

            # Reference: Best-of-N
            "best_of_n_acc": float(best_correct / num_questions) if num_questions > 0 else 0.0,
            "best_of_n_acc_gpt": float(best_correct_gpt / num_questions) if num_questions > 0 else 0.0,

            "query_latency": f"{(time.time()-t_start)/len(all_finished)*1000:.0f} ms"
        }

        print(f"\n{'='*60}")
        print(f"Results for {src_file}:")
        print(f"  Rollout: {rollout_num}x")
        print(f"  Questions: {num_questions}")
        print(f"  Majority Voting Acc: {overall_results['majority_voting_acc']:.4f}")
        print(f"  Majority Voting Acc GPT: {overall_results['majority_voting_acc_gpt']:.4f}")
        print(f"  Average Acc: {overall_results['average_acc']:.4f}")
        print(f"  Best-of-N Acc: {overall_results['best_of_n_acc']:.4f}")
        print(f"{'='*60}\n")

        final_metrics = {"overall": overall_results}

        # ==== Save results ====
        t = time.localtime()
        split = "test"
        result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.rollout{rollout_num}.json'
        metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.rollout{rollout_num}.metrics.json'

        # Pick the output dir based on dataset name in src_file
        if "math500" in src_file:
            output_dir = f'../outputs/math500/{model_short_name}.cot'
        elif "amc23" in src_file:
            output_dir = f'../outputs/amc23/{model_short_name}.cot'
        elif "amc22" in src_file:
            output_dir = f'../outputs/amc22/{model_short_name}.cot'
        elif "minervamath" in src_file:
            output_dir = f'../outputs/minervamath/{model_short_name}.cot'
        elif "olympiad" in src_file:
            output_dir = f'../outputs/olympiad/{model_short_name}.cot'
        elif "aime24" in src_file:
            output_dir = f'../outputs/aime24/{model_short_name}.cot'
        elif "aime25" in src_file:
            output_dir = f'../outputs/aime25/{model_short_name}.cot'
        elif "v4_train" in src_file:
            output_dir = f'../outputs/v4_train/{model_short_name}.cot'
        elif "math_all" in src_file:
            output_dir = f'../outputs/math_all/{model_short_name}.cot'
        else:
            output_dir = f'../outputs/other/{model_short_name}.cot'
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, result_json_name), "w", encoding="utf-8") as f:
            json.dump(all_finished, f, indent=4, ensure_ascii=False)

        with open(os.path.join(output_dir, metrics_json_name), "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=4, ensure_ascii=False)


        print(f"Results saved to {output_dir}")
        print(json.dumps(final_metrics, indent=2))

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
