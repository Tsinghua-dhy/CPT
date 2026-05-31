#!/usr/bin/env python3
"""
Split pairs_data_all_judged.jsonl consensus items into four parts: 
1. 10000 items for SFT
2. 10000 items for RL
3. 500 items for Test
4. the remainder

Only consensus items (has_consensus == True) are kept.
"""

import json
import random
from pathlib import Path
from tqdm import tqdm


def split_consensus_data(input_file, output_dir):
    """
    Split the consensus data.
    Args:
        input_file: inputfilepath
        output_dir: output directory
    """
    # Read all consensus items
    print(f"Reading data from {input_file}...")
    print("Filtering consensus data only...")
    
    all_data = []
    total_count = 0
    consensus_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading"), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total_count += 1
                
                # Keep only consensus items with a non-empty final_judgment
                judge_results = data.get('judge_results', {})
                has_consensus = judge_results.get('has_consensus', False)
                final_judgment = judge_results.get('final_judgment', '')
                
                # Filter out items with empty judgments
                if has_consensus and final_judgment and final_judgment.strip() != '':
                    all_data.append(data)
                    consensus_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"\nWarning: Skipping line {line_num} due to JSON error: {e}")
                continue
    
    print(f"\nTotal pairs: {total_count:,}")
    print(f"Consensus pairs: {consensus_count:,} ({consensus_count/total_count*100:.2f}%)")
    print(f"Non-consensus pairs: {total_count - consensus_count:,}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_data)
    
    # Split
    sft_count = 10000
    rl_count = 10000
    test_count = 500
    
    sft_data = all_data[:sft_count]
    rl_data = all_data[sft_count:sft_count + rl_count]
    test_data = all_data[sft_count + rl_count:sft_count + rl_count + test_count]
    remaining_data = all_data[sft_count + rl_count + test_count:]
    
    print(f"\nData split:")
    print(f"  SFT: {len(sft_data):,} pairs")
    print(f"  RL: {len(rl_data):,} pairs")
    print(f"  Test: {len(test_data):,} pairs")
    print(f"  Remaining: {len(remaining_data):,} pairs")
    
    # Make sure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # saveSFTdata
    sft_file = output_dir / 'sft_data.jsonl'
    print(f"\nSaving SFT data to {sft_file}...")
    with open(sft_file, 'w', encoding='utf-8') as f:
        for item in tqdm(sft_data, desc="Writing SFT"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # saveRLdata
    rl_file = output_dir / 'rl_data.jsonl'
    print(f"Saving RL data to {rl_file}...")
    with open(rl_file, 'w', encoding='utf-8') as f:
        for item in tqdm(rl_data, desc="Writing RL"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # saveTestdata
    test_file = output_dir / 'test_data.jsonl'
    print(f"Saving Test data to {test_file}...")
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in tqdm(test_data, desc="Writing Test"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save the remaining data
    remaining_file = output_dir / 'remaining_data.jsonl'
    print(f"Saving remaining data to {remaining_file}...")
    with open(remaining_file, 'w', encoding='utf-8') as f:
        for item in tqdm(remaining_data, desc="Writing Remaining"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Compute statistics
    stats = {
        'source_file': str(input_file),
        'total_pairs': total_count,
        'consensus_pairs': consensus_count,
        'non_consensus_pairs': total_count - consensus_count,
        'consensus_rate': f"{consensus_count/total_count:.2%}",
        'sft_pairs': len(sft_data),
        'rl_pairs': len(rl_data),
        'test_pairs': len(test_data),
        'remaining_pairs': len(remaining_data),
        'split_ratio': {
            'sft': f"{len(sft_data)/consensus_count:.2%}",
            'rl': f"{len(rl_data)/consensus_count:.2%}",
            'test': f"{len(test_data)/consensus_count:.2%}",
            'remaining': f"{len(remaining_data)/consensus_count:.2%}"
        }
    }
    
    # Append detailed stats
    def get_detailed_stats(data, name):
        """Compute detailed dataset statistics."""
        difficulty_dist = {}
        pair_type_dist = {}
        task_type_dist = {}
        judgment_difficulty_dist = {}
        
        for item in data:
            # Difficulty distribution
            difficulty = item.get('difficulty', 'unknown')
            difficulty_dist[difficulty] = difficulty_dist.get(difficulty, 0) + 1
            
            # Pair-type distribution
            pair_type = item.get('pair_type', 'unknown')
            pair_type_dist[pair_type] = pair_type_dist.get(pair_type, 0) + 1
            
            # Task-type distribution
            task_type = item.get('task_type', 'unknown')
            task_type_dist[task_type] = task_type_dist.get(task_type, 0) + 1
            
            # Judgement-difficulty distribution
            judge_results = item.get('judge_results', {})
            judgment_difficulty = judge_results.get('judgment_difficulty', 'unknown')
            judgment_difficulty_dist[judgment_difficulty] = judgment_difficulty_dist.get(judgment_difficulty, 0) + 1
        
        return {
            'count': len(data),
            'difficulty': difficulty_dist,
            'pair_type': pair_type_dist,
            'task_type': task_type_dist,
            'judgment_difficulty': judgment_difficulty_dist
        }
    
    stats['sft_details'] = get_detailed_stats(sft_data, 'SFT')
    stats['rl_details'] = get_detailed_stats(rl_data, 'RL')
    stats['test_details'] = get_detailed_stats(test_data, 'Test')
    stats['remaining_details'] = get_detailed_stats(remaining_data, 'Remaining')
    
    stats_file = output_dir / 'split_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("Split complete!")
    print(f"{'='*80}")
    print(f"SFT data: {sft_file}")
    print(f"RL data: {rl_file}")
    print(f"Test data: {test_file}")
    print(f"Remaining data: {remaining_file}")
    print(f"Statistics: {stats_file}")
    print(f"{'='*80}")
    
    return stats


if __name__ == '__main__':
    input_file = './judged_pairs/pairs_data_all_judged.jsonl'
    output_dir = './post_training'
    
    stats = split_consensus_data(input_file, output_dir)
    
    print("\n" + "="*80)
    print("Split Statistics Summary")
    print("="*80)
    print(f"Source: {stats['source_file']}")
    print(f"\nOriginal Data:")
    print(f"  Total pairs: {stats['total_pairs']:,}")
    print(f"  Consensus pairs: {stats['consensus_pairs']:,} ({stats['consensus_rate']})")
    print(f"  Non-consensus pairs: {stats['non_consensus_pairs']:,}")
    print(f"\nSplit Results:")
    print(f"  SFT: {stats['sft_pairs']:,} ({stats['split_ratio']['sft']})")
    print(f"  RL: {stats['rl_pairs']:,} ({stats['split_ratio']['rl']})")
    print(f"  Test: {stats['test_pairs']:,} ({stats['split_ratio']['test']})")
    print(f"  Remaining: {stats['remaining_pairs']:,} ({stats['split_ratio']['remaining']})")
    print("="*80)
