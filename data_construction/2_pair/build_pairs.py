#!/usr/bin/env python3
"""
Pair-construction script for the judged_pairs dataset (mix_math_data version).

Task definitions (all tasks construct pairs grouped by question): 
- task_1: intra-model pairing
  * easy/medium: 1 pair per question
  * hard/very_hard: 2 pairs per question
  
- task_2: inter-model pairing
  * easy/medium: 1 pair per question
  * hard/very_hard: 2 pairs per question
  
- task_3: small modelvslarge model
  * medium/hard/very_hard: 1 pair per qualifying question
  * condition: the question must have both small-model and large-model rollouts, with the small model getting at least 1 right and the large model at least 1 wrong

Truncated rollouts (no answer) are normalized to pred = "be truncated, No answer".
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import random


def normalize_rollout(rollout: Dict) -> Dict:
    """
    Normalize a rollout, handling truncation cases.
    """
    normalized = rollout.copy()
    
    # If pred_ans is empty, treat as truncated
    if not normalized.get('pred_ans', '').strip():
        normalized['pred_ans'] = 'be truncated, No answer'
    
    return normalized


def load_all_questions(output_dir: Path) -> Dict:
    """
    Load all rollouts and group them by question text.
    return: {
        question_text: {
            'difficulty': str,
            'question': str,
            'answer': str,
            'question_idx': int,  # keep one question_idx for output
            'rollouts': {model_name: [rollouts]}
        }
    }
    """
    all_questions = {}
    
    # Iterate over each model directory
    for model_dir in output_dir.glob('*'):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Find this model's JSON files (skip metrics files and shards) 
        json_files = [f for f in model_dir.glob('*.json') 
                     if 'metrics' not in f.name and 'shard' not in f.name]
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                # Use the question text as the unique key
                q_text = item['question']
                
                # First time we see this question -> initialize
                if q_text not in all_questions:
                    all_questions[q_text] = {
                        'difficulty': item['difficulty'],
                        'question': item['question'],
                        'answer': item['answer'],
                        'question_idx': item['question_idx'],  # keep one idx
                        'rollouts': defaultdict(list)
                    }
                
                # Normalize the rollout and tag it with the model name
                normalized_item = normalize_rollout(item)
                normalized_item['model_name'] = model_name
                
                # Append to this question's rollouts
                all_questions[q_text]['rollouts'][model_name].append(normalized_item)
    
    return all_questions


def get_model_size(model_name: str) -> str:
    """
    Derive the model size from the model name.
    """
    if '14b' in model_name.lower():
        return '14b'
    elif '8b' in model_name.lower():
        return '8b'
    elif '4b' in model_name.lower():
        return '4b'
    return 'unknown'


def create_pair(rollout_1: Dict, rollout_2: Dict, pair_type: str, task_type: str, 
                question: str, answer: str, difficulty: str, q_idx: int) -> Dict:
    """
    Build the data structure for one pair of rollouts.
    """
    # Extract model info
    model_name_1 = rollout_1.get('model_name', 'unknown')
    model_name_2 = rollout_2.get('model_name', 'unknown')
    
    # Randomly decide which rollout becomes path A vs path B
    if random.random() < 0.5:
        path_a, path_b = rollout_1, rollout_2
        path_a_name, path_b_name = model_name_1, model_name_2
    else:
        path_a, path_b = rollout_2, rollout_1
        path_a_name, path_b_name = model_name_2, model_name_1
    
    return {
        'question': question,
        'ground_truth': answer,
        'difficulty': difficulty,
        'question_idx': q_idx,
        'pair_type': pair_type,
        'task_type': task_type,
        
        'path_a': {
            'model_name': path_a_name,
            'model_size': get_model_size(path_a_name),
            'rollout_idx': path_a['rollout_idx'],
            'pred_ans': path_a['pred_ans'],
            'gen_text': path_a['gen_text_store'],
            'is_correct_gpt': path_a['Metrics']['math_equal_gpt'],
        },
        'path_b': {
            'model_name': path_b_name,
            'model_size': get_model_size(path_b_name),
            'rollout_idx': path_b['rollout_idx'],
            'pred_ans': path_b['pred_ans'],
            'gen_text': path_b['gen_text_store'],
            'is_correct_gpt': path_b['Metrics']['math_equal_gpt'],
        },
    }


def weighted_sample_rollouts(rollouts: List[Dict], k: int = 2) -> List[Dict]:
    """
    Weighted sampling over rollouts.
    - correct rollouts have weight 3
    - wrong rollouts have weight 2
    """
    if len(rollouts) < k:
        return []
    
    # Compute the weight per rollout
    weights = []
    for r in rollouts:
        if r['Metrics']['math_equal_gpt'] == 1:
            weights.append(3)  # correct -> weight 3
        else:
            weights.append(2)  # wrong -> weight 2
    
    # Use random.choices for weighted sampling (with replacement) 
    # Then make sure the chosen rollouts are distinct
    selected_indices = set()
    attempts = 0
    max_attempts = 100
    
    while len(selected_indices) < k and attempts < max_attempts:
        idx = random.choices(range(len(rollouts)), weights=weights, k=1)[0]
        selected_indices.add(idx)
        attempts += 1
    
    if len(selected_indices) < k:
        # If we cannot pick k distinct rollouts, fall  afterwards to plain sampling
        return random.sample(rollouts, k)
    
    return [rollouts[i] for i in selected_indices]


def task_1_intra_model_pairs(all_questions: Dict) -> List[Dict]:
    """
    Task 1: build intra-model pairs
    - easy/medium: 1 pair per question
    - hard/very_hard: 2 pairs per question
    - Weighted sampling: correct answers get weight 3, wrong answers get weight 2
    - Ensure pairs from the same question are non-duplicated
    """
    pairs = []
    
    # Per-difficulty count of pairs to generate per question
    pairs_per_question = {
        'easy': 1,
        'medium': 1,
        'hard': 2,
        'very_hard': 2,
    }
    
    # Iterate over every question
    for q_idx, q_data in all_questions.items():
        difficulty = q_data['difficulty']
        num_pairs = pairs_per_question.get(difficulty, 1)
        
        # Iterate over each model
        for model_name, rollouts in q_data['rollouts'].items():
            # This model needs >= 2 rollouts on this question to form a pair
            if len(rollouts) < 2:
                continue
            
            # Track pairs we've already produced (key = frozenset of rollout indices) 
            used_pairs = set()
            
            # Generate the requested number of pairs for this (question, model)
            attempts = 0
            max_attempts = num_pairs * 50  # max number of retries
            
            while len(used_pairs) < num_pairs and attempts < max_attempts:
                attempts += 1
                
                # Pick two distinct rollouts via weighted random sampling
                selected = weighted_sample_rollouts(rollouts, k=2)
                if len(selected) < 2:
                    continue
                
                # Build a canonical pair id (frozenset of rollout indices, order-insensitive) 
                pair_key = frozenset([selected[0]['rollout_idx'], selected[1]['rollout_idx']])
                
                # Skip if this pair was already produced
                if pair_key in used_pairs:
                    continue
                
                used_pairs.add(pair_key)
                
                pair = create_pair(
                    selected[0], selected[1],
                    pair_type='intra_model',
                    task_type='task_1',
                    question=q_data['question'],
                    answer=q_data['answer'],
                    difficulty=difficulty,
                    q_idx=q_idx
                )
                pairs.append(pair)
    
    return pairs


def task_2_inter_model_pairs(all_questions: Dict) -> List[Dict]:
    """
    Task 2: build inter-model pairs
    - easy: 1 pair per (question, model_pair)
    - medium: 1 pair per (question, model_pair)
    - hard: 2 pairs per (question, model_pair)
    - very_hard: 2 pairs per (question, model_pair)
    
    Note: iterate all unordered model combinations (positional bias removed). 
    With models {A, B} there is only one combination: (A, B).
    Ensure pairs from the same question are non-duplicated.
    """
    pairs = []
    
    # Per-difficulty count of pairs per (question, model-pair)
    pairs_per_question = {
        'easy': 1,
        'medium': 1,
        'hard': 2,
        'very_hard': 2,
    }
    
    # Collect all model names
    all_model_names = set()
    for q_data in all_questions.values():
        all_model_names.update(q_data['rollouts'].keys())
    
    all_model_names = sorted(list(all_model_names))
    
    # Build all unordered model combinations (combinations, not permutations) 
    # e.g. for two models {A, B} -> (A, B)
    from itertools import combinations
    model_pairs = list(combinations(all_model_names, 2))
    
    print(f"Info: Task 2 model combinations (unordered): {len(model_pairs)}")
    for m1, m2 in model_pairs:
        print(f"  - ({m1}, {m2})")
    
    # Iterate over every question
    for q_idx, q_data in all_questions.items():
        difficulty = q_data['difficulty']
        num_pairs = pairs_per_question.get(difficulty, 1)
        rollouts_by_model = q_data['rollouts']
        
        # Iterate over each model pair
        for model1, model2 in model_pairs:
            # Check whether this question has rollouts from BOTH models
            if model1 not in rollouts_by_model or model2 not in rollouts_by_model:
                continue
            
            # Track pairs we've already produced (key = tuple of two rollout indices) 
            used_pairs = set()
            
            # Generate the requested number of pairs for this (question, model-pair)
            attempts = 0
            max_attempts = num_pairs * 50  # max number of retries
            
            while len(used_pairs) < num_pairs and attempts < max_attempts:
                attempts += 1
                
                # Pick rollouts via weighted sampling
                r1_list = weighted_sample_rollouts(rollouts_by_model[model1], k=1)
                r2_list = weighted_sample_rollouts(rollouts_by_model[model2], k=1)
                
                if not r1_list or not r2_list:
                    continue
                
                r1 = r1_list[0]
                r2 = r2_list[0]
                
                # Build a canonical pair id (tuple of two rollout indices) 
                # Note: model1 and model2 are ordered here, so a tuple is fine
                pair_key = (r1['rollout_idx'], r2['rollout_idx'])
                
                # Skip if this pair was already produced
                if pair_key in used_pairs:
                    continue
                
                used_pairs.add(pair_key)
                
                # Bucket by whether the two answers agree
                if r1['pred_ans'] == r2['pred_ans'] and r1['pred_ans'] != 'be truncated, No answer':
                    pair_type = 'inter_model_same'
                else:
                    pair_type = 'inter_model_diff'
                
                pair = create_pair(
                    r1, r2,
                    pair_type=pair_type,
                    task_type='task_2',
                    question=q_data['question'],
                    answer=q_data['answer'],
                    difficulty=difficulty,
                    q_idx=q_data['question_idx']
                )
                pairs.append(pair)
    
    return pairs


def task_3_small_vs_large_pairs(all_questions: Dict) -> List[Dict]:
    """
    Task 3: small modelvslarge model
    - medium/hard/very_hard: 1 pair per qualifying question
    - condition: the question has rollouts from both a small and a large model, with the small one having >= 1 correct and the large one having >= 1 wrong
    - All possible small-vs-large combinations: 4B vs 8B, 4B vs 14B, 8B vs 14B
    """
    pairs = []
    
    # Collect all model names
    all_model_names = set()
    for q_data in all_questions.values():
        all_model_names.update(q_data['rollouts'].keys())
    
    # Group models by size
    models_by_size = {
        '4b': [],
        '8b': [],
        '14b': []
    }
    
    for model_name in all_model_names:
        size = get_model_size(model_name)
        if size in models_by_size:
            models_by_size[size].append(model_name)
    
    # Define every small-vs-large combination (small relative to large) 
    # Format: (small_model_size, large_model_size, pair_type_label)
    size_combinations = []
    
    if models_by_size['4b'] and models_by_size['8b']:
        size_combinations.append(('4b', '8b', '4b_vs_8b'))
    if models_by_size['4b'] and models_by_size['14b']:
        size_combinations.append(('4b', '14b', '4b_vs_14b'))
    if models_by_size['8b'] and models_by_size['14b']:
        size_combinations.append(('8b', '14b', '8b_vs_14b'))
    
    if not size_combinations:
        print("Warning: No valid small-large model combinations found for task_3")
        return pairs
    
    print(f"Info: Task 3 will generate pairs for {len(size_combinations)} combinations:")
    for small_size, large_size, label in size_combinations:
        print(f"  - {small_size} (small) vs {large_size} (large)")
    
    # For each combination, iterate over every question
    for small_size, large_size, pair_type_label in size_combinations:
        small_model_list = models_by_size[small_size]
        large_model_list = models_by_size[large_size]
        
        combination_pairs = 0
        
        for q_idx, q_data in all_questions.items():
            difficulty = q_data['difficulty']
            
            # Skip easy; handle only medium/hard/very_hard
            if difficulty not in ['medium', 'hard', 'very_hard']:
                continue
            
            rollouts_by_model = q_data['rollouts']
            
            # Check whether the question has rollouts from BOTH the small and large model
            has_small = any(model in rollouts_by_model for model in small_model_list)
            has_large = any(model in rollouts_by_model for model in large_model_list)
            
            if not (has_small and has_large):
                continue
            
            # Find a CORRECT rollout from the small model
            small_correct = []
            for small_model in small_model_list:
                if small_model in rollouts_by_model:
                    small_correct.extend([r for r in rollouts_by_model[small_model] 
                                         if r['Metrics']['math_equal_gpt'] == 1])
            
            # Find a WRONG rollout from the large model
            large_incorrect = []
            for large_model in large_model_list:
                if large_model in rollouts_by_model:
                    large_incorrect.extend([r for r in rollouts_by_model[large_model] 
                                           if r['Metrics']['math_equal_gpt'] == 0])
            
            # If both are found, build a pair
            if small_correct and large_incorrect:
                r1 = random.choice(small_correct)
                r2 = random.choice(large_incorrect)
                
                pair = create_pair(
                    r1, r2,
                    pair_type=f'small_correct_large_incorrect_{pair_type_label}',
                    task_type='task_3',
                    question=q_data['question'],
                    answer=q_data['answer'],
                    difficulty=difficulty,
                    q_idx=q_idx
                )
                pairs.append(pair)
                combination_pairs += 1
        
        print(f"  {pair_type_label}: generated {combination_pairs} pairs")
    
    return pairs


def generate_statistics(pairs: List[Dict]) -> Dict:
    """
    Compute summary statistics.
    """
    stats = {
        'total_pairs': len(pairs),
        'by_task': defaultdict(int),
        'by_difficulty': defaultdict(int),
        'by_pair_type': defaultdict(int),
        'truncated_count': 0,
        'correct_pairs': 0,
        'incorrect_pairs': 0,
        'mixed_pairs': 0,
        # Per-task correctness stats
        'by_task_correctness': {
            'task_1': {'both_correct': 0, 'both_incorrect': 0, 'mixed': 0},
            'task_2': {'both_correct': 0, 'both_incorrect': 0, 'mixed': 0},
            'task_3': {'both_correct': 0, 'both_incorrect': 0, 'mixed': 0},
        }
    }
    
    for pair in pairs:
        task = pair['task_type']
        stats['by_task'][task] += 1
        stats['by_difficulty'][pair['difficulty']] += 1
        stats['by_pair_type'][pair['pair_type']] += 1
        
        # Check for truncation
        if (pair['path_a']['pred_ans'] == 'be truncated, No answer' or 
            pair['path_b']['pred_ans'] == 'be truncated, No answer'):
            stats['truncated_count'] += 1
        
        ra_correct = pair['path_a']['is_correct_gpt']
        rb_correct = pair['path_b']['is_correct_gpt']
        
        # Overall stats
        if ra_correct and rb_correct:
            stats['correct_pairs'] += 1
            stats['by_task_correctness'][task]['both_correct'] += 1
        elif not ra_correct and not rb_correct:
            stats['incorrect_pairs'] += 1
            stats['by_task_correctness'][task]['both_incorrect'] += 1
        else:
            stats['mixed_pairs'] += 1
            stats['by_task_correctness'][task]['mixed'] += 1
    
    return dict(stats)


def analyze_answer_distribution(pairs: List[Dict]) -> Dict:
    """
    Analyse the distribution of agreeing-answer vs. disagreeing-answer pairs.
    """
    analysis = {
        'task_1': {'same': 0, 'diff': 0},
        'task_2': {'same': 0, 'diff': 0},
        'task_3': {'same': 0, 'diff': 0},
        'overall': {'same': 0, 'diff': 0},
    }
    
    for pair in pairs:
        task = pair['task_type']
        
        # Compare the two answers
        if pair['path_a']['pred_ans'] == pair['path_b']['pred_ans']:
            analysis[task]['same'] += 1
            analysis['overall']['same'] += 1
        else:
            analysis[task]['diff'] += 1
            analysis['overall']['diff'] += 1
    
    return analysis


def generate_readme(output_dir: Path, stats: Dict, answer_dist: Dict, total_pairs: int):
    """
    generate READMEfile
    """
    readme_path = output_dir / 'README.md'
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# Mix Math Data - Judged Pairs Dataset\n\n")
        f.write("## Overview\n\n")
        f.write("This dataset builds reasoning-trace pairs from multi-model rollouts, used to train downstream judges. \n\n")
        
        f.write("## Task definition\n\n")
        f.write("### Task 1: intra-model pairs\n")
        f.write("- easy/medium: 1 pair per question\n")
        f.write("- hard/very_hard: 2 pairs per question\n\n")
        
        f.write("### Task 2: inter-model pairs\n")
        f.write("- easy/medium: 1 pair per question\n")
        f.write("- hard/very_hard: 2 pairs per question\n\n")
        
        f.write("### Task 3: small modelvslarge model\n")
        f.write("- medium/hard/very_hard: 1 pair per qualifying question\n")
        f.write("- condition: the question has rollouts from BOTH a small and a large model, with the small one having at least 1 correct and the large one at least 1 wrong\n\n")
        
        f.write("## Statistics\n\n")
        f.write(f"- total pairs: {total_pairs}\n")
        f.write(f"- pairs containing truncated rollouts: {stats.get('truncated_count', 0)}\n\n")
        
        f.write("### By task type\n\n")
        for task, count in sorted(stats['by_task'].items()):
            f.write(f"- {task}: {count}\n")
        
        f.write("\n### By difficulty\n\n")
        for diff, count in sorted(stats['by_difficulty'].items()):
            f.write(f"- {diff}: {count}\n")
        
        f.write("\n### By pair type\n\n")
        for pair_type, count in sorted(stats['by_pair_type'].items()):
            f.write(f"- {pair_type}: {count}\n")
        
        f.write("\n### By correctness\n\n")
        f.write(f"- both correct: {stats.get('correct_pairs', 0)}\n")
        f.write(f"- both wrong: {stats.get('incorrect_pairs', 0)}\n")
        f.write(f"- one correct, one wrong: {stats.get('mixed_pairs', 0)}\n")
        
        f.write("\n### Answer distribution (same vs different)\n\n")
        for task in ['task_1', 'task_2', 'task_3', 'overall']:
            same = answer_dist[task]['same']
            diff = answer_dist[task]['diff']
            total = same + diff
            if total > 0:
                same_pct = same / total * 100
                diff_pct = diff / total * 100
                f.write(f"**{task}**:\n")
                f.write(f"- same answer: {same} ({same_pct:.1f}%)\n")
                f.write(f"- different answer: {diff} ({diff_pct:.1f}%)\n\n")
        
        f.write("\n## Files\n\n")
        f.write("- `pairs_data.jsonl`: all pair data\n")
        f.write("- `statistics.json`: detailed statistics\n")
        f.write("- `README.md`: this file\n")


def main():
    
    # pathconfig
    output_dir = Path('./outputs')
    result_dir = Path('./judged_pairs')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Mix Math Data - Judged Pairs Generation")
    print("="*80)
    
    # Load all rollouts grouped by question
    print("\nLoading rollouts grouped by question...")
    all_questions = load_all_questions(output_dir)
    print(f"find  {len(all_questions)} question")
    
    # Per-model question counts
    model_question_counts = defaultdict(int)
    for q_data in all_questions.values():
        for model_name in q_data['rollouts'].keys():
            model_question_counts[model_name] += 1
    
    print("\nQuestions covered by each model:")
    for model_name, count in sorted(model_question_counts.items()):
        print(f"  - {model_name} ({get_model_size(model_name)}): {count} questions")
    
    # Execute the three tasks
    print("\nRunning Task 1: intra-model pairs...")
    task_1_pairs = task_1_intra_model_pairs(all_questions)
    print(f"  generated {len(task_1_pairs)} pairs")
    
    print("\nRunning Task 2: inter-model pairs...")
    task_2_pairs = task_2_inter_model_pairs(all_questions)
    print(f"  generated {len(task_2_pairs)} pairs")
    
    print("\nexecute Task 3: small modelvslarge model...")
    task_3_pairs = task_3_small_vs_large_pairs(all_questions)
    print(f"  generated {len(task_3_pairs)} pairs")
    
    # Per-difficulty counts for Task 3
    task_3_by_diff = defaultdict(int)
    for pair in task_3_pairs:
        task_3_by_diff[pair['difficulty']] += 1
    print("  Task 3 per-difficulty distribution:")
    for diff in ['medium', 'hard', 'very_hard']:
        print(f"    - {diff}: {task_3_by_diff[diff]} pairs")
    
    # Merge all pairs
    all_pairs = task_1_pairs + task_2_pairs + task_3_pairs
    print(f"\nGenerated {len(all_pairs)} pairs in total")
    
    # savedata
    pairs_file = result_dir / 'pairs_data.jsonl'
    with open(pairs_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\nWrote pair data to: {pairs_file}")
    
    # Compute statistics
    stats = generate_statistics(all_pairs)
    answer_dist = analyze_answer_distribution(all_pairs)
    
    # Print detailed stats
    print("\n" + "="*80)
    print("Detailed statistics")
    print("="*80)
    
    print("\nBy task type:")
    for task in ['task_1', 'task_2', 'task_3']:
        task_stats = stats['by_task_correctness'][task]
        total = stats['by_task'][task]
        print(f"\n{task}: {total} pairs total")
        print(f"  both correct: {task_stats['both_correct']} ({task_stats['both_correct']/total*100:.1f}%)")
        print(f"  both wrong: {task_stats['both_incorrect']} ({task_stats['both_incorrect']/total*100:.1f}%)")
        print(f"  one correct, one wrong: {task_stats['mixed']} ({task_stats['mixed']/total*100:.1f}%)")
    
    print(f"\nOverall statistics:")
    print(f"total pairs: {stats['total_pairs']}")
    print(f"  both correct: {stats['correct_pairs']} ({stats['correct_pairs']/stats['total_pairs']*100:.1f}%)")
    print(f"  both wrong: {stats['incorrect_pairs']} ({stats['incorrect_pairs']/stats['total_pairs']*100:.1f}%)")
    print(f"  one correct, one wrong: {stats['mixed_pairs']} ({stats['mixed_pairs']/stats['total_pairs']*100:.1f}%)")
    
    # Print answer distribution
    print("\n" + "="*80)
    print("Answer distribution (same vs different)")
    print("="*80)
    for task in ['task_1', 'task_2', 'task_3', 'overall']:
        same = answer_dist[task]['same']
        diff = answer_dist[task]['diff']
        total = same + diff
        if total > 0:
            same_pct = same / total * 100
            diff_pct = diff / total * 100
            print(f"\n{task}:")
            print(f"  same answer: {same} ({same_pct:.1f}%)")
            print(f"  different answer: {diff} ({diff_pct:.1f}%)")
            print(f"  ratio: {same_pct:.1f}:{diff_pct:.1f}")
    
    stats_file = result_dir / 'statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nWrote statistics to: {stats_file}")
    
    # generate README
    generate_readme(result_dir, stats, answer_dist, len(all_pairs))
    print(f"Wrote README to: {result_dir / 'README.md'}")
    
    print("\n" + "="*80)
    print("done!")
    print("="*80)


if __name__ == "__main__":
    main()
