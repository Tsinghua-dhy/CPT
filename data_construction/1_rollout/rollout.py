import argparse
from collections import defaultdict
import copy
import json
import os
import random
import re
import time

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Import evaluation helpers
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'eval', 'utils'))
from math_equal import eval_math_with_gpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--difficulty", type=str, default=None, help="Process specific difficulty: easy, medium, hard, very_hard")
    parser.add_argument("--output_suffix", type=str, default="", help="Output file suffix")
    parser.add_argument("--shard_id", type=int, default=None, help="Shard ID: 0 or 1 for splitting data in half")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    return parser.parse_args()


def process_text(example, tokenizer):
    """Build the prompt."""
    question_raw = example["question"]
    prompt = f"""The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The final answer should be enclosed within \\boxed{{}}.

User:{question_raw}
Assistant:"""
    example["chat_prompt"] = prompt
    return example


def extract_answer_math(s):
    """extractanswer"""
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
    print("=" * 80)
    print("DAPOMath Rollout Evaluation")
    print("=" * 80)
    
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    t_start = time.time()
    
    model_path = args.model_path
    model_short_name = model_path.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_gpus = torch.cuda.device_count()
    
    print(f"\nModel: {model_short_name}")
    print(f"GPUs: {num_gpus} ({args.gpu_id})")
    print(f"Source: {args.src_file}")
    
    # initialize vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=args.gpu_memory_rate,
        trust_remote_code=True
    )

    # Read items and bucket by difficulty
    print("\nLoading data...")
    data_by_difficulty = defaultdict(list)
    with open(args.src_file, "r") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            difficulty = obj.get("difficulty", "unknown")
            # If a difficulty was specified, only load that difficulty
            if args.difficulty is None or difficulty == args.difficulty:
                data_by_difficulty[difficulty].append(obj)
            if args.max_samples > 0 and i >= args.max_samples - 1:
                break
    
    print("\nData distribution:")
    for diff in ['easy', 'medium', 'hard', 'very_hard']:
        count = len(data_by_difficulty[diff])
        if count > 0:
            print(f"  {diff}: {count}")
    
    # Per-difficulty rollout count and max_tokens
    # Detect the model from its name (largest first, since '14b' contains '4b') 
    if '14b' in model_short_name.lower():
        rollout_config = {
            'easy': {'rollout': 2, 'max_tokens': 2048},
            'medium': {'rollout': 2, 'max_tokens': 2048},
            'hard': {'rollout': 4, 'max_tokens': 4096},
            'very_hard': {'rollout': 4, 'max_tokens': 4096}
        }
        model_size = '14b'
    elif '8b' in model_short_name.lower():
        rollout_config = {
            'easy': {'rollout': 4, 'max_tokens': 2048},
            'medium': {'rollout': 4, 'max_tokens': 2048},
            'hard': {'rollout': 8, 'max_tokens': 4096},
            'very_hard': {'rollout': 8, 'max_tokens': 4096}
        }
        model_size = '8b'
    elif '4b' in model_short_name.lower():
        rollout_config = {
            'easy': {'rollout': 8, 'max_tokens': 2048},
            'medium': {'rollout': 8, 'max_tokens': 2048},
            'hard': {'rollout': 16, 'max_tokens': 4096},
            'very_hard': {'rollout': 16, 'max_tokens': 4096}
        }
        model_size = '4b'
    else:
        print("Warning: Cannot determine model size, using default config")
        rollout_config = {
            'easy': {'rollout': 4, 'max_tokens': 2048},
            'medium': {'rollout': 4, 'max_tokens': 2048},
            'hard': {'rollout': 8, 'max_tokens': 4096},
            'very_hard': {'rollout': 8, 'max_tokens': 4096}
        }
        model_size = 'unknown'
    
    print(f"\nModel size detected: {model_size}")
    print("Rollout configuration:")
    for diff, config in rollout_config.items():
        print(f"  {diff}: {config['rollout']}x rollout, max_tokens={config['max_tokens']}")
    
    # Output buffer for all rollouts
    all_results = []
    question_rollouts = defaultdict(list)
    global_question_idx = 0
    
    # Process by difficulty
    for difficulty in ['easy', 'medium', 'hard', 'very_hard']:
        data_ori = data_by_difficulty[difficulty]
        if len(data_ori) == 0:
            continue
        
        # If a shard was specified, only process that shard
        if args.shard_id is not None and args.num_shards > 1:
            shard_size = len(data_ori) // args.num_shards
            start_idx = args.shard_id * shard_size
            if args.shard_id == args.num_shards - 1:
                # The last shard absorbs all remaining items
                end_idx = len(data_ori)
            else:
                end_idx = start_idx + shard_size
            data_ori = data_ori[start_idx:end_idx]
            print(f"Shard {args.shard_id}/{args.num_shards}: processing indices {start_idx}-{end_idx}")
        
        config = rollout_config[difficulty]
        rollout_num = config['rollout']
        max_tokens = config['max_tokens']
        
        print(f"\n{'='*80}")
        print(f"Processing {difficulty} ({len(data_ori)} questions, {rollout_num}x rollout, max_tokens={max_tokens})")
        print(f"{'='*80}")
        
        # Prepare items
        data = [copy.deepcopy(item) for item in data_ori]
        for item in data:
            process_text(item, tokenizer)
        
        # Build all prompts
        prompts = []
        prompt_to_question_map = []
        for i, item in enumerate(data):
            for rollout_idx in range(rollout_num):
                prompts.append(item["chat_prompt"])
                prompt_to_question_map.append((global_question_idx + i, rollout_idx))
        
        print(f"Generating {len(prompts)} outputs ({len(data)} questions × {rollout_num} rollouts)...")
        
        # generate 
        sampling_params = SamplingParams(
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        outputs = llm.generate(prompts, sampling_params)
        
        # parseresult
        for output_idx, output in enumerate(outputs):
            q_idx, rollout_idx = prompt_to_question_map[output_idx]
            local_idx = q_idx - global_question_idx
            
            q = data[local_idx]["question"]
            a = data[local_idx]["answer"]
            gen_text = output.outputs[0].text.strip()
            pred = extract_answer_math(gen_text)
            
            # If no answer was extracted, mark as empty
            if pred == "":
                pred = ""  # leave empty; the downstream evaluation handles it
            
            question_rollouts[q_idx].append({
                "question": q,
                "answer": a,
                "pred_ans": pred,
                "gen_text_store": gen_text,
                "difficulty": difficulty,
            })
        
        global_question_idx += len(data)
    
    print(f"\n{'='*80}")
    print(f"Generation completed: {len(question_rollouts)} questions")
    print(f"{'='*80}")
    
    # batchevaluation
    print("\nEvaluating all rollouts...")
    
    all_golden_answers = []
    all_pred_answers = []
    all_questions = []
    eval_map = []
    
    for q_idx in sorted(question_rollouts.keys()):
        rollouts = question_rollouts[q_idx]
        for rollout_idx, rollout in enumerate(rollouts):
            all_golden_answers.append(rollout["answer"])
            all_pred_answers.append(rollout["pred_ans"])
            all_questions.append(rollout["question"])
            eval_map.append((q_idx, rollout_idx))
    
    print(f"Total evaluations: {len(all_golden_answers)}")
    
    # batchevaluation
    eval_results = eval_math_with_gpt(all_golden_answers, all_pred_answers, all_questions)
    
    # Distribute the results
    for idx, (q_idx, rollout_idx) in enumerate(eval_map):
        question_rollouts[q_idx][rollout_idx]["is_equal"] = eval_results[idx]["is_equal"]
        question_rollouts[q_idx][rollout_idx]["is_equal_gpt"] = eval_results[idx]["is_equal_gpt"]
    
    # Compute metrics
    print("\nComputing metrics...")
    
    majority_correct = 0
    majority_correct_gpt = 0
    all_correct = []
    all_correct_gpt = []
    best_correct = 0
    best_correct_gpt = 0
    
    # Per-difficulty stats
    difficulty_stats = defaultdict(lambda: {
        'total': 0,
        'majority_correct': 0,
        'majority_correct_gpt': 0,
        'all_correct': [],
        'all_correct_gpt': [],
        'best_correct': 0,
        'best_correct_gpt': 0
    })
    
    all_finished = []
    
    for q_idx in sorted(question_rollouts.keys()):
        rollouts = question_rollouts[q_idx]
        difficulty = rollouts[0]["difficulty"]
        
        is_equal_list = [r["is_equal"] for r in rollouts]
        is_equal_gpt_list = [r["is_equal_gpt"] for r in rollouts]
        
        # Majority Voting
        majority_vote = 1 if sum(is_equal_list) > len(is_equal_list) / 2 else 0
        majority_vote_gpt = 1 if sum(is_equal_gpt_list) > len(is_equal_gpt_list) / 2 else 0
        majority_correct += majority_vote
        majority_correct_gpt += majority_vote_gpt
        
        # Average
        all_correct.extend(is_equal_list)
        all_correct_gpt.extend(is_equal_gpt_list)
        
        # Best-of-N
        best_correct += (1 if any(is_equal_list) else 0)
        best_correct_gpt += (1 if any(is_equal_gpt_list) else 0)
        
        # Per-difficulty stats
        difficulty_stats[difficulty]['total'] += 1
        difficulty_stats[difficulty]['majority_correct'] += majority_vote
        difficulty_stats[difficulty]['majority_correct_gpt'] += majority_vote_gpt
        difficulty_stats[difficulty]['all_correct'].extend(is_equal_list)
        difficulty_stats[difficulty]['all_correct_gpt'].extend(is_equal_gpt_list)
        difficulty_stats[difficulty]['best_correct'] += (1 if any(is_equal_list) else 0)
        difficulty_stats[difficulty]['best_correct_gpt'] += (1 if any(is_equal_gpt_list) else 0)
        
        # Save detailed results
        for rollout_idx, rollout in enumerate(rollouts):
            all_finished.append({
                "question_idx": q_idx,
                "rollout_idx": rollout_idx,
                "difficulty": difficulty,
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
    
    # overallresult
    overall_results = {
        "model_size": model_size,
        "num_questions": num_questions,
        "total_generations": len(all_finished),
        
        "majority_voting_acc": float(majority_correct / num_questions) if num_questions > 0 else 0.0,
        "majority_voting_acc_gpt": float(majority_correct_gpt / num_questions) if num_questions > 0 else 0.0,
        
        "average_acc": float(np.mean(all_correct)) if len(all_correct) > 0 else 0.0,
        "average_acc_gpt": float(np.mean(all_correct_gpt)) if len(all_correct_gpt) > 0 else 0.0,
        
        "best_of_n_acc": float(best_correct / num_questions) if num_questions > 0 else 0.0,
        "best_of_n_acc_gpt": float(best_correct_gpt / num_questions) if num_questions > 0 else 0.0,
        
        "query_latency": f"{(time.time()-t_start)/len(all_finished)*1000:.0f} ms"
    }
    
    # Per-difficulty results
    difficulty_results = {}
    for diff in ['easy', 'medium', 'hard', 'very_hard']:
        stats = difficulty_stats[diff]
        if stats['total'] > 0:
            difficulty_results[diff] = {
                "num_questions": stats['total'],
                "rollout_num": rollout_config[diff]['rollout'],
                "max_tokens": rollout_config[diff]['max_tokens'],
                "majority_voting_acc": float(stats['majority_correct'] / stats['total']),
                "majority_voting_acc_gpt": float(stats['majority_correct_gpt'] / stats['total']),
                "average_acc": float(np.mean(stats['all_correct'])) if len(stats['all_correct']) > 0 else 0.0,
                "average_acc_gpt": float(np.mean(stats['all_correct_gpt'])) if len(stats['all_correct_gpt']) > 0 else 0.0,
                "best_of_n_acc": float(stats['best_correct'] / stats['total']),
                "best_of_n_acc_gpt": float(stats['best_correct_gpt'] / stats['total']),
            }
    
    print(f"\n{'='*80}")
    print("Overall Results:")
    print(f"  Model: {model_short_name} ({model_size})")
    print(f"  Questions: {num_questions}")
    print(f"  Majority Voting Acc: {overall_results['majority_voting_acc']:.4f}")
    print(f"  Majority Voting Acc GPT: {overall_results['majority_voting_acc_gpt']:.4f}")
    print(f"  Average Acc: {overall_results['average_acc']:.4f}")
    print(f"  Best-of-N Acc: {overall_results['best_of_n_acc']:.4f}")
    
    print(f"\nResults by Difficulty:")
    for diff in ['easy', 'medium', 'hard', 'very_hard']:
        if diff in difficulty_results:
            res = difficulty_results[diff]
            print(f"  {diff}:")
            print(f"    Questions: {res['num_questions']}, Rollout: {res['rollout_num']}x")
            print(f"    Majority Voting: {res['majority_voting_acc']:.4f}")
            print(f"    Average: {res['average_acc']:.4f}")
            print(f"    Best-of-N: {res['best_of_n_acc']:.4f}")
    print(f"{'='*80}")
    
    final_metrics = {
        "overall": overall_results,
        "by_difficulty": difficulty_results
    }
    
    # saveresult
    t = time.localtime()
    output_dir = f'outputs/dapomath/{model_short_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = f".{args.output_suffix}" if args.output_suffix else ""
    result_json_name = f'test.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{model_size}{suffix}.json'
    metrics_json_name = f'test.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{model_size}{suffix}.metrics.json'
    
    with open(os.path.join(output_dir, result_json_name), "w", encoding="utf-8") as f:
        json.dump(all_finished, f, indent=4, ensure_ascii=False)
    
    with open(os.path.join(output_dir, metrics_json_name), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - {result_json_name}")
    print(f"  - {metrics_json_name}")


if __name__ == "__main__":
    main()
