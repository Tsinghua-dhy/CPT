#!/usr/bin/env python3
"""
AbstentionBench evaluation script (v2)

Unified evaluation pipeline. Supports:
1. multi-GPU parallel inference
2. manydatasetevaluation
3. two prompt styles: normal vs. abstention-encouraging
4. LLM-as-judge abstention detection
5. three correctness-evaluation modes:
   - math: math_verify + GPT fallback
   - multiple-choice: string match
   - open-ended: LLM-as-Judge

Important: vLLM cannot fully release GPU memory within a single process
(confirmed upstream; see GitHub issues #1908, #5211, #6544, #18806). 
Therefore Phase 1 (vLLM inference) runs in a subprocess. When the subprocess
exits, the OS reclaims all GPU resources automatically. 
Phase 2 (GPT-API evaluation) runs in the main process and does not touch the GPU. 
"""

import argparse
import copy
import json
import os
import re
import sys
import time
import tempfile
import subprocess
from collections import defaultdict
from typing import List, Dict, Optional

# Make eval/utils/ importable so we can reuse `prompts.py` etc.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

import gc


# ============================================================
# Valid datasets (20) 
# ============================================================
VALID_DATASETS = {
    "alcuna",
    "bbq", 
    "big_bench_disambiguate",
    "big_bench_known_unknowns",
    "coconot",
    "falseqa",
    "gpqa_abstain",
    "gsm8k_abstain",
    "known_unknown_questions",
    "mediq",
    "mmlu_history_abstain",
    "mmlu_math_abstain",
    "moral_choice",
    "musique",
    "qaqa",
    "qasper",
    "situated_qa",
    "squad2",
    "umwp",
    "world_sense",
    "all",  # special value: union of all datasets
}

# Valid prompt styles
VALID_PROMPT_STYLES = {"normal", "abstention"}


def validate_datasets(dataset_str: str) -> List[str]:
    """Validate the dataset name."""
    datasets = [d.strip() for d in dataset_str.split(',')]
    invalid_datasets = []
    
    for ds in datasets:
        if ds not in VALID_DATASETS:
            invalid_datasets.append(ds)
    
    if invalid_datasets:
        raise ValueError(
            f"Invalid dataset name(s): {invalid_datasets}\n"
            f"Valid datasets are: {sorted(VALID_DATASETS)}\n"
        )
    
    return datasets


def validate_prompt_styles(style_str: str) -> List[str]:
    """Validate the prompt style."""
    styles = [s.strip() for s in style_str.split(',')]
    invalid_styles = []
    
    for style in styles:
        if style not in VALID_PROMPT_STYLES:
            invalid_styles.append(style)
    
    if invalid_styles:
        raise ValueError(
            f"Invalid prompt style(s): {invalid_styles}\n"
            f"Valid styles are: {sorted(VALID_PROMPT_STYLES)}"
        )
    
    return styles


def parse_args():
    parser = argparse.ArgumentParser(description="AbstentionBench Evaluation V2")
    
    # Basic args
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3",
                        help="GPU IDs to use, e.g., '0,1,2,3'")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95,
                        help="GPU memory utilization rate")
    
    # datasetparameter
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset to evaluate. Use 'all' for all datasets, or specify subset name")
    parser.add_argument("--dataset_dir", type=str, 
                        default="./abstention_bench",
                        help="Directory containing dataset files")
    parser.add_argument("--src_file", type=str, default=None,
                        help="Specific source file to evaluate (overrides --dataset)")
    
    # Prompt style
    parser.add_argument("--prompt_style", type=str, default="normal",
                        help="Prompt style(s), comma-separated: 'normal', 'abstention', or 'normal,abstention'")
    parser.add_argument("--normal_prompt_variants", type=str, default="",
                        help="Comma-separated normal_prompt variant ids (e.g. 'v0,v1,v2,v3,v4,v5'). "
                             "When non-empty and prompt_style includes 'normal', all variants run within a single model load. "
                             "Per-variant outputs go under {output_dir}/{variant}/{model_name}.normal/. "
                             "Empty means a single run, controlled by env NORMAL_PROMPT_VARIANT (default v0).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip variants whose metrics already exist under output_dir (checked via metrics.json).")
    
    # generationparameter
    parser.add_argument("--temp", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max tokens to generate")
    parser.add_argument("--rollout_num", type=int, default=1,
                        help="Number of rollouts per question")
    
    # evaluationparameter
    parser.add_argument("--use_llm_judge", action="store_true",
                        help="Use LLM as a judge for abstention detection")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Force using chat template for instruct models")
    parser.add_argument("--skip_correctness", action="store_true",
                        help="Skip correctness evaluation (only evaluate abstention)")
    parser.add_argument("--max_threads", type=int, default=256,
                        help="GPT API concurrency for abstention LLM judge / correctness evaluator (default 256).")
    
    # outputparameter
    parser.add_argument("--output_dir", type=str, default="../outputs/abstention_bench",
                        help="Output directory")
    parser.add_argument("--output_prefix", type=str, default="",
                        help="Prefix for output files")
    
    # Sampling args
    parser.add_argument("--start_sample", type=int, default=0,
                        help="Start sample index")
    parser.add_argument("--end_sample", type=int, default=100000,
                        help="End sample index")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to evaluate (0 for all)")
    parser.add_argument("--max_per_dataset", type=int, default=3500,
                        help="Max samples per dataset when using 'all' (default: 3500, same as original AbstentionBench paper; 0 for no limit)")
    parser.add_argument("--indices_file", type=str, 
                        default="./abstention_bench/raw_data/subsampling-indices.json",
                        help="Path to official subsampling-indices.json from AbstentionBench")
    
    # Internal flags (set by the subprocess; do not pass manually) 
    parser.add_argument("--_inference_subprocess", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--_inference_output_file", type=str, default=None,
                        help=argparse.SUPPRESS)
    
    return parser.parse_args()


def detect_model_type(model_name: str) -> str:
    """Detect the model type.
    
    Returns:
        'it-thinking': instruct + thinking model (e.g. qwen3-8b-it-thinking) , 
                       uses system+user chat template; vLLM enables thinking by default.
        'it': instruct model (e.g. qwen3-4b-instruct / it) , 
              uses system+user chat template, no thinking.
        'base': base model, uses plain User-Assistant text format.
    """
    model_name_lower = model_name.lower()
    
    # Match it-thinking first (must come before plain it, since it-thinking contains 'it') 
    if 'it-thinking' in model_name_lower:
        return 'it-thinking'
    
    # Detect plain instruct model
    instruct_keywords = ['instruct', '-it', '_it']
    for keyword in instruct_keywords:
        if keyword in model_name_lower:
            return 'it'
    
    # Check whether the path contains an `instruct` keyword
    # e.g. /models/qwen3-4b-instruct/... 
    if 'instruct' in model_name_lower or 'chat' in model_name_lower:
        return 'it'
    
    return 'base'


def is_instruct_model(model_name: str) -> bool:
    """Auto-detect if the model is an instruct model (backward compatibility)"""
    return detect_model_type(model_name) in ('it', 'it-thinking')


def load_dataset(dataset_dir: str, dataset_name: str, max_per_dataset: int = 0) -> List[Dict]:
    """Load dataset from jsonl file
    
    Args:
        dataset_dir: Directory containing dataset files
        dataset_name: Name of dataset or 'all' for merged
        max_per_dataset: Max samples per dataset (only used when dataset_name='all', 0 for no limit)
    """
    if dataset_name == "all":
        file_path = os.path.join(dataset_dir, "all_merged.jsonl")
    else:
        file_path = os.path.join(dataset_dir, f"{dataset_name}.jsonl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # When dataset='all' and max_per_dataset is set, sub-sample each dataset uniformly.
    if dataset_name == "all" and max_per_dataset > 0:
        data = balanced_sample_from_merged(data, max_per_dataset)
    
    return data


# Dataset name map (internal name -> name used in the official indices file) 
# Hoisted to a module-level constant for reuse.
DATASET_NAME_MAPPING = {
    "alcuna": "ALCUNADataset",
    "bbq": "BBQDataset", 
    "known_unknown_questions": "KUQDataset",
    "squad2": "Squad2Dataset",
}

# Cache the official indices to avoid re-loading
_cached_official_indices = None
_cached_indices_file = None

def load_official_indices(indices_file: str) -> Dict:
    """Load the official sub-sampling indices (cached)."""
    global _cached_official_indices, _cached_indices_file
    
    if _cached_official_indices is not None and _cached_indices_file == indices_file:
        return _cached_official_indices
    
    if indices_file and os.path.exists(indices_file):
        with open(indices_file, 'r') as f:
            _cached_official_indices = json.load(f)
        _cached_indices_file = indices_file
        print(f"Loaded official subsampling indices from: {indices_file}")
    else:
        _cached_official_indices = {}
    
    return _cached_official_indices


def sample_single_dataset(data: List[Dict], dataset_name: str, max_samples: int, indices_file: str = None) -> List[Dict]:
    """Sub-sample a single dataset following the AbstentionBench paper.

    Datasets larger than ``max_samples`` (paper default 3500) are sub-sampled
    using the official indices file; smaller datasets are returned unchanged.

    Args:
        data: items of a single dataset.
        dataset_name: dataset name.
        max_samples: maximum number of items to keep (paper default 3500).
        indices_file: path to the official ``subsampling-indices.json``.

    Returns:
        the sub-sampled item list.
    """
    original_count = len(data)
    
    if max_samples <= 0 or original_count <= max_samples:
        # No sub-sampling needed
        print(f"  {dataset_name}: {original_count} items (kept as-is)")
        return data
    
    # Sub-sample
    official_indices = load_official_indices(indices_file)
    official_key = DATASET_NAME_MAPPING.get(dataset_name)
    
    if official_key and official_key in official_indices:
        # Use the official indices
        indices_to_keep = set(official_indices[official_key])
        sampled = [item for i, item in enumerate(data) if i in indices_to_keep]
        print(f"  {dataset_name}: {original_count} -> {len(sampled)} (official indices)")
    else:
        # No official indices: deterministically take the first `max_samples` items 
        sampled = data[:max_samples]
        print(f"  {dataset_name}: {original_count} -> {len(sampled)} (first {max_samples} items)")
    
    return sampled


def balanced_sample_from_merged(data: List[Dict], max_per_dataset: int, indices_file: str = None) -> List[Dict]:
    """Sub-sample each constituent dataset uniformly from a merged corpus.

    Follows the AbstentionBench paper protocol:
    - datasets larger than ``max_per_dataset`` (default 3500) are sub-sampled
      via the official indices file;
    - smaller datasets are kept as-is.

    Args:
        data: merged item list.
        max_per_dataset: per-dataset cap (paper default 3500).
        indices_file: path to the official ``subsampling-indices.json``.

    Returns:
        the balanced sub-sampled item list.
    """
    from collections import defaultdict
    
    # Group by dataset and keep each item's within-dataset index
    dataset_groups = defaultdict(list)
    dataset_local_idx = defaultdict(int)
    
    for item in data:
        ds_name = item.get("dataset", "unknown")
        local_idx = dataset_local_idx[ds_name]
        dataset_groups[ds_name].append((local_idx, item))
        dataset_local_idx[ds_name] += 1
    
    # Original distribution
    print(f"\nOriginal dataset distribution (total {len(data)} items):")
    total_original = 0
    for ds_name, items in sorted(dataset_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {ds_name}: {len(items)}")
        total_original += len(items)
    
    # Load the official indices
    official_indices = load_official_indices(indices_file)
    
    # Sub-sample each dataset
    sampled_data = []
    print(f"\nSub-sampled distribution (cap {max_per_dataset} per dataset, AbstentionBench protocol):")
    total_sampled = 0
    
    for ds_name, items in sorted(dataset_groups.items()):
        original_count = len(items)
        
        if original_count <= max_per_dataset:
            # No sub-sampling needed, keep all items 
            sampled = [item for _, item in items]
            print(f"  {ds_name}: {original_count} (kept as-is)")
        else:
            # Sub-sample, preferring the official indices
            official_key = DATASET_NAME_MAPPING.get(ds_name)
            
            if official_key and official_key in official_indices:
                # Use the official indices
                indices_to_keep = set(official_indices[official_key])
                sampled = [item for local_idx, item in items if local_idx in indices_to_keep]
                print(f"  {ds_name}: {original_count} -> {len(sampled)} (official indices)")
            else:
                # No official indices: deterministically take the first `max_per_dataset` items 
                sampled = [item for _, item in items[:max_per_dataset]]
                print(f"  {ds_name}: {original_count} -> {len(sampled)} (first {max_per_dataset} items)")
        
        sampled_data.extend(sampled)
        total_sampled += len(sampled)
    
    print(f"\nSub-sampled total: {total_sampled} items (from {total_original})")
    
    return sampled_data


def extract_answer(response: str, dataset_name: str = "", question: str = "") -> str:
    """
    Extract the answer span depending on dataset type.
    """
    from correctness_evaluator import (
        get_dataset_type,
        extract_math_answer,
        extract_choice_answer,
        extract_boxed_answer
    )
    from correctness_evaluator import MULTIPLE_CHOICE_DATASETS
    
    dataset_type = get_dataset_type(dataset_name)
    
    if dataset_type == "math":
        return extract_math_answer(response)
    elif dataset_type == "multiple_choice":
        # Multiple-choice: try \boxed{} letter first
        boxed = extract_boxed_answer(response)
        if boxed and len(boxed) == 1 and boxed.upper() in 'ABCDEFGHIJ':
            return boxed.upper()
        # Otherwise fall back to the choice extractor
        choice = extract_choice_answer(response)
        if choice:
            return choice
    
    # Default: \boxed{} content or first sentence
    boxed = extract_boxed_answer(response)
    if boxed:
        return boxed
    
    sentences = response.split('.')
    if sentences:
        return sentences[0].strip()[:200]
    
    return response.strip()[:200]


# ============================================================
# Phase 1: subprocess inference entry
# ============================================================
def run_inference_subprocess(args):
    """
    Run vLLM inference in a subprocess. 
    
    When the subprocess exits the OS reclaims all GPU memory. 
    This is the only reliable GPU-memory release scheme accepted by the vLLM community. 
    (Refs: vLLM GitHub issues #1908, #5211, #6544, #18806)
    """
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from correctness_evaluator import MULTIPLE_CHOICE_DATASETS
    from prompts import (
        get_normal_prompt_messages,
        get_normal_prompt_text,
        get_normal_prompt_messages_instruct,
        get_abstention_prompt_messages,
        get_abstention_prompt_text,
        get_abstention_prompt_messages_instruct,
    )
    
    def process_text_inner(example, tokenizer, prompt_style, use_chat_template, model_type):
        """Process a single example to create prompt"""
        question = example.get("question", "")
        dataset_name = example.get("dataset", "")
        is_multiple_choice = dataset_name in MULTIPLE_CHOICE_DATASETS
        
        if use_chat_template and model_type in ('it', 'it-thinking'):
            if prompt_style == "abstention":
                messages = get_abstention_prompt_messages_instruct(question, is_multiple_choice=is_multiple_choice)
            else:
                messages = get_normal_prompt_messages_instruct(question, is_multiple_choice=is_multiple_choice)
            
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False,
            }
            if model_type == 'it-thinking':
                template_kwargs["enable_thinking"] = True
            elif model_type == 'it':
                template_kwargs["enable_thinking"] = False
            
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        elif use_chat_template:
            if prompt_style == "abstention":
                messages = get_abstention_prompt_messages(question)
            else:
                messages = get_normal_prompt_messages(question)
            
            prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
        else:
            if prompt_style == "abstention":
                prompt = get_abstention_prompt_text(question, is_multiple_choice=is_multiple_choice)
            else:
                prompt = get_normal_prompt_text(question, is_multiple_choice=is_multiple_choice)
        
        example["chat_prompt"] = prompt
        return example
    
    # Validate args
    prompt_styles = validate_prompt_styles(args.prompt_style)
    
    if not args.src_file:
        dataset_names = validate_datasets(args.dataset)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Load tokenizer and model
    print(f"\n[Subprocess] Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    num_gpus = torch.cuda.device_count()
    print(f"[Subprocess] Using {num_gpus} GPUs: {args.gpu_id}")
    
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=args.gpu_memory_rate,
        trust_remote_code=True
    )
    
    # Determine model type and chat template usage
    model_type = detect_model_type(args.model_path)
    use_chat_template = args.use_chat_template or model_type in ('it', 'it-thinking')
    
    print(f"[Subprocess] Model type: {model_type}")
    print(f"[Subprocess] Prompt styles: {prompt_styles}")
    print(f"[Subprocess] Use chat template: {use_chat_template}")
    if model_type == 'it-thinking':
        print(f"[Subprocess] Thinking mode: enabled")
    elif model_type == 'it':
        print(f"[Subprocess] Thinking mode: disabled")
    
    # Get datasets to evaluate
    if args.src_file:
        src_files = [args.src_file]
    else:
        if args.dataset == "all":
            src_files = [os.path.join(args.dataset_dir, "all_merged.jsonl")]
        else:
            src_files = [
                os.path.join(args.dataset_dir, f"{name.strip()}.jsonl")
                for name in dataset_names
            ]
    
    # ================================================================
    # Phase 1: inference
    # ================================================================
    inference_cache = []

    for src_file in src_files:
        if not os.path.exists(src_file):
            print(f"[Subprocess] Warning: File not found, skipping: {src_file}")
            continue
        
        is_all_merged = "all_merged" in src_file or args.dataset == "all"
        dataset_name_from_file = os.path.basename(src_file).replace('.jsonl', '')
        
        data_ori = []
        with open(src_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if args.start_sample <= i < args.end_sample:
                    if line.strip():
                        data_ori.append(json.loads(line))
                if args.max_samples > 0 and len(data_ori) >= args.max_samples:
                    break
        
        if args.max_per_dataset > 0:
            original_count = len(data_ori)
            
            if is_all_merged:
                from collections import Counter
                ds_counts = Counter(item.get("dataset", "unknown") for item in data_ori)
                max_count = max(ds_counts.values()) if ds_counts else 0
                
                if max_count <= args.max_per_dataset:
                    print(f"[Subprocess] data already pre-sampled, largest dataset has {max_count} items, skipping run-time sampling")
                else:
                    print(f"[Subprocess] detected large dataset (max {max_count} items), running run-time sampling...")
                    data_ori = balanced_sample_from_merged(data_ori, args.max_per_dataset, args.indices_file)
            else:
                print(f"\n[Subprocess] sub-sampling a single dataset (max_per_dataset={args.max_per_dataset}):")
                data_ori = sample_single_dataset(
                    data_ori, 
                    dataset_name_from_file, 
                    args.max_per_dataset, 
                    args.indices_file
                )
        
        if not data_ori:
            print(f"[Subprocess] No data to evaluate for {src_file}, skipping")
            continue
        
        for prompt_style in prompt_styles:
            # Decide which variants to run under this prompt_style
            # - normal: if normal_prompt_variants is set, run them sequentially
            # - other styles: run once with empty variant_id
            if prompt_style == "normal" and args.normal_prompt_variants.strip():
                variant_ids = [v.strip() for v in args.normal_prompt_variants.split(",") if v.strip()]
            else:
                variant_ids = [""]

            for variant_id in variant_ids:
                # select the prompt variant via env vars; consumed by prompts.get_normal_prompt_text
                if variant_id:
                    os.environ["NORMAL_PROMPT_VARIANT"] = variant_id
                tag = f"{prompt_style}" + (f"/{variant_id}" if variant_id else "")

                print(f"\n{'='*80}")
                print(f"[Subprocess - Inference] {src_file}")
                print(f"Prompt style: {tag}")
                print(f"{'='*80}")

                print(f"[Subprocess] Loaded {len(data_ori)} samples")

                data = []
                for item in data_ori:
                    processed = process_text_inner(
                        copy.deepcopy(item),
                        tokenizer,
                        prompt_style,
                        use_chat_template,
                        model_type=model_type
                    )
                    data.append(processed)

                print(f"\n[Subprocess] Generating responses (rollout={args.rollout_num})...")

                all_prompts = []
                prompt_map = []

                for i, item in enumerate(data):
                    for r in range(args.rollout_num):
                        all_prompts.append(item["chat_prompt"])
                        prompt_map.append((i, r))

                stop_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]

                sampling_params = SamplingParams(
                    temperature=args.temp,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    stop=stop_tokens
                )

                t_start = time.time()
                outputs = llm.generate(all_prompts, sampling_params)
                t_gen = time.time() - t_start

                print(f"[Subprocess] Generation time: {t_gen:.2f}s ({len(all_prompts)} prompts)")

                question_responses = defaultdict(list)

                for output_idx, output in enumerate(outputs):
                    sample_idx, rollout_idx = prompt_map[output_idx]
                    response = output.outputs[0].text.strip()
                    dataset_name = data[sample_idx].get("dataset", "")

                    question_responses[sample_idx].append({
                        "rollout_idx": rollout_idx,
                        "response": response,
                        "extracted_answer": extract_answer(response, dataset_name, data[sample_idx].get("question", ""))
                    })

                # Drop chat_prompt to reduce serialization size
                data_for_cache = []
                for item in data:
                    item_copy = {k: v for k, v in item.items() if k != "chat_prompt"}
                    data_for_cache.append(item_copy)

                # Convert defaultdict -> dict and int keys -> str (required by JSON) 
                qr_serializable = {str(k): v for k, v in question_responses.items()}

                inference_cache.append({
                    "src_file": src_file,
                    "prompt_style": prompt_style,
                    "variant_id": variant_id,
                    "is_all_merged": is_all_merged,
                    "data": data_for_cache,
                    "question_responses": qr_serializable,
                })

                print(f"[Subprocess] Inference cached for {os.path.basename(src_file)} ({tag})")
    
    # Save inference results to disk
    output_file = args._inference_output_file
    print(f"\n[Subprocess] Saving inference results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(inference_cache, f, ensure_ascii=False)
    
    print(f"[Subprocess] Inference complete. Exiting subprocess (GPU will be freed by OS).")
    # Subprocess exits cleanly; OS reclaims all GPU memory


def main():
    args = parse_args()
    
    # ================================================================
    # In subprocess mode, run inference and exit
    # ================================================================
    if args._inference_subprocess:
        run_inference_subprocess(args)
        return
    
    # ================================================================
    # Main-process logic
    # ================================================================
    print("=" * 80)
    print("AbstentionBench Evaluation V2")
    print("=" * 80)
    print(f"Evaluation method by dataset type:")
    print(f"  - Math datasets: math_verify + GPT fallback")
    print(f"  - Multiple choice: string matching")
    print(f"  - Other: LLM-as-Judge")
    print(f"GPU memory release: subprocess isolation (vLLM official recommendation)")
    print("=" * 80)
    
    # Validate args
    try:
        prompt_styles = validate_prompt_styles(args.prompt_style)
    except ValueError as e:
        print(f"\nError: {e}")
        return
    
    if not args.src_file:
        try:
            dataset_names = validate_datasets(args.dataset)
        except ValueError as e:
            print(f"\nError: {e}")
            return

    # ================================================================
    # skip_existing: when enabled, skip variants whose metrics already exist under output_dir,
    # under output_dir; if all variants are already done, exit early.
    # ================================================================
    def _model_subdir_name(model_path: str) -> str:
        parts = model_path.rstrip('/').split('/')
        if len(parts) >= 2:
            return parts[-2] + '_' + parts[-1]
        return parts[-1] if parts else 'unknown_model'

    if args.skip_existing and args.normal_prompt_variants.strip() and "normal" in prompt_styles:
        requested = [v.strip() for v in args.normal_prompt_variants.split(",") if v.strip()]
        remaining = []
        skipped = []
        model_name_for_check = _model_subdir_name(args.model_path)
        for vid in requested:
            # Artifact path: {output_dir}/{variant}/{model_name}.normal/*all_merged*.metrics.json
            variant_dir = os.path.join(args.output_dir, vid, f"{model_name_for_check}.normal")
            has_any_metrics = False
            if os.path.isdir(variant_dir):
                for fn in os.listdir(variant_dir):
                    if fn.endswith(".metrics.json") and "all_merged" in fn:
                        has_any_metrics = True
                        break
            if has_any_metrics:
                skipped.append(vid)
            else:
                remaining.append(vid)
        if skipped:
            print(f"[skip_existing] already existsartifact, skip variants: {skipped}")
        if not remaining:
            print("[skip_existing] all requested variants already done, exiting.")
            return
        args.normal_prompt_variants = ",".join(remaining)
        print(f"[skip_existing] variants to run: {remaining}")

    # ================================================================
    # Phase 1: run vLLM inference in a subprocess
    # ================================================================
    print(f"\n{'='*80}")
    print("[Phase 1 - Inference] Launching inference subprocess...")
    print(f"{'='*80}")
    
    # Create temp files for inference results
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', prefix='abstention_inference_',
        dir=args.output_dir if os.path.isdir(args.output_dir) else '/tmp',
        delete=False
    ) as tmp_f:
        inference_output_file = tmp_f.name
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build the subprocess command: same script + --_inference_subprocess marker
    cmd = [sys.executable, os.path.abspath(__file__)]
    cmd += ["--_inference_subprocess"]
    cmd += ["--_inference_output_file", inference_output_file]
    cmd += ["--model_path", args.model_path]
    cmd += ["--gpu_id", args.gpu_id]
    cmd += ["--gpu_memory_rate", str(args.gpu_memory_rate)]
    cmd += ["--dataset", args.dataset]
    cmd += ["--dataset_dir", args.dataset_dir]
    if args.src_file:
        cmd += ["--src_file", args.src_file]
    cmd += ["--prompt_style", args.prompt_style]
    if args.normal_prompt_variants:
        cmd += ["--normal_prompt_variants", args.normal_prompt_variants]
    cmd += ["--temp", str(args.temp)]
    cmd += ["--top_p", str(args.top_p)]
    cmd += ["--top_k", str(args.top_k)]
    cmd += ["--max_tokens", str(args.max_tokens)]
    cmd += ["--rollout_num", str(args.rollout_num)]
    if args.use_chat_template:
        cmd += ["--use_chat_template"]
    cmd += ["--start_sample", str(args.start_sample)]
    cmd += ["--end_sample", str(args.end_sample)]
    cmd += ["--max_samples", str(args.max_samples)]
    cmd += ["--max_per_dataset", str(args.max_per_dataset)]
    cmd += ["--indices_file", args.indices_file]
    cmd += ["--max_threads", str(args.max_threads)]
    
    print(f"[Phase 1] Subprocess command: {' '.join(cmd[:6])} ...")
    
    t_phase1_start = time.time()
    
    # Run subprocess and forward stdout in real time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu_id}
    )
    
    # Real-time stream of subprocess output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    t_phase1 = time.time() - t_phase1_start
    
    if process.returncode != 0:
        print(f"\nERROR: Inference subprocess failed with return code {process.returncode}")
        # Clean up temp files
        if os.path.exists(inference_output_file):
            os.remove(inference_output_file)
        return
    
    print(f"\n[Phase 1] Inference subprocess completed in {t_phase1:.1f}s")
    print(f"[Phase 1] GPU memory has been FULLY RELEASED (subprocess exited)")
    
    # Read inference results
    print(f"\n[Phase 1] Loading inference results from: {inference_output_file}")
    with open(inference_output_file, 'r', encoding='utf-8') as f:
        inference_cache = json.load(f)
    
    # Clean up temp files
    os.remove(inference_output_file)
    print(f"[Phase 1] Loaded {len(inference_cache)} inference tasks")
    
    # Restore int keys on question_responses
    for cache_entry in inference_cache:
        qr = cache_entry["question_responses"]
        cache_entry["question_responses"] = {int(k): v for k, v in qr.items()}
    
    # ================================================================
    # Phase 2: abstention detection + correctness evaluation + save (CPU + GPT API only) 
    # ================================================================
    # Deferred import to avoid unnecessary CUDA init in the main process 
    from abstention_detector import (
        batch_detect_abstention,
        calculate_abstention_metrics,
    )
    from correctness_evaluator import (
        batch_evaluate_correctness,
        get_dataset_type,
    )
    
    print(f"\n{'='*80}")
    print(f"[Phase 2 - Evaluation] Starting evaluation for {len(inference_cache)} tasks...")
    print(f"[Phase 2] This phase uses only GPT API calls (NO GPU needed)")
    print(f"{'='*80}")
    
    all_results = {}
    
    for cache_entry in inference_cache:
        src_file = cache_entry["src_file"]
        prompt_style = cache_entry["prompt_style"]
        variant_id = cache_entry.get("variant_id", "")
        is_all_merged = cache_entry["is_all_merged"]
        data = cache_entry["data"]
        question_responses = cache_entry["question_responses"]
        
        print(f"\n{'='*80}")
        print(f"[Phase 2 - Evaluation] {os.path.basename(src_file)} ({prompt_style})")
        print(f"{'='*80}")
        
        # Abstention detection
        print("\nDetecting abstention...")
        
        all_items_for_detection = []
        for i in sorted(question_responses.keys()):
            item = data[i]
            for resp in question_responses[i]:
                all_items_for_detection.append({
                    "question": item.get("question", ""),
                    "response": resp["response"]
                })
        
        abstention_results = batch_detect_abstention(
            all_items_for_detection,
            use_llm=args.use_llm_judge,
            max_threads=args.max_threads,
        )
        
        # Assign abstention results back
        result_idx = 0
        for i in sorted(question_responses.keys()):
            for resp in question_responses[i]:
                resp["abstention_detection"] = abstention_results[result_idx]
                result_idx += 1
        
        # Calculate metrics
        print("\nCalculating metrics...")
        
        predictions = []
        gold_labels = []
        detailed_results = []
        
        for i in sorted(question_responses.keys()):
            item = data[i]
            responses = question_responses[i]
            
            abstention_votes = [
                r["abstention_detection"]["is_abstention"] 
                for r in responses
            ]
            is_abstention = sum(abstention_votes) > len(abstention_votes) / 2
            
            predictions.append({"is_abstention": is_abstention})
            gold_labels.append(item.get("should_abstain", False))
            
            detailed_results.append({
                "question": item.get("question", ""),
                "reference_answers": item.get("reference_answers"),
                "should_abstain": item.get("should_abstain", False),
                "dataset": item.get("dataset", "unknown"),
                "responses": responses,
                "predicted_abstention": is_abstention,
                "abstention_method": responses[0]["abstention_detection"]["method"] if responses else "none"
            })
        
        metrics = calculate_abstention_metrics(predictions, gold_labels)
        
        # ============================================================
        # Correctness evaluation v2: dispatch by dataset type
        # ============================================================
        if not args.skip_correctness:
            print("\nEvaluating correctness (V2 - dataset-aware)...")
            
            items_for_correctness = []
            correctness_indices = []
            denominator_indices = []
            
            for i, result in enumerate(detailed_results):
                if not result["should_abstain"] and result["reference_answers"]:
                    denominator_indices.append(i)
                    predicted_abstention = result.get("predicted_abstention", False)
                    if not predicted_abstention:
                        dataset_type = get_dataset_type(result["dataset"])
                        if dataset_type == "llm_judge":
                            response_for_eval = result["responses"][0]["response"] if result["responses"] else ""
                        else:
                            response_for_eval = result["responses"][0]["extracted_answer"] if result["responses"] else ""
                        items_for_correctness.append({
                            "question": result["question"],
                            "reference_answers": result["reference_answers"],
                            "response": response_for_eval,
                            "dataset": result["dataset"]
                        })
                        correctness_indices.append(i)
            
            correctness_map = {}
            if items_for_correctness:
                correctness_results = batch_evaluate_correctness(
                    items_for_correctness,
                    question_key="question",
                    reference_key="reference_answers",
                    response_key="response",
                    dataset_key="dataset",
                    max_threads=args.max_threads,
                )
                for idx, (orig_idx, corr_result) in enumerate(zip(correctness_indices, correctness_results)):
                    correctness_map[orig_idx] = corr_result
                    detailed_results[orig_idx]["correctness_result"] = corr_result
            
            if denominator_indices:
                math_correct, math_total = 0, 0
                choice_correct, choice_total = 0, 0
                llm_judge_correct, llm_judge_total = 0, 0
                
                for orig_idx in denominator_indices:
                    dataset_name = detailed_results[orig_idx]["dataset"]
                    dataset_type = get_dataset_type(dataset_name)
                    
                    if orig_idx in correctness_map:
                        is_correct = correctness_map[orig_idx].get("is_correct", False)
                    else:
                        is_correct = False
                        detailed_results[orig_idx]["correctness_result"] = {
                            "is_correct": False,
                            "reason": "model_abstained_so_not_evaluated"
                        }
                    
                    if dataset_type == "math":
                        math_total += 1
                        if is_correct:
                            math_correct += 1
                    elif dataset_type == "multiple_choice":
                        choice_total += 1
                        if is_correct:
                            choice_correct += 1
                    else:
                        llm_judge_total += 1
                        if is_correct:
                            llm_judge_correct += 1
                
                total_correct = math_correct + choice_correct + llm_judge_correct
                total = math_total + choice_total + llm_judge_total
                
                metrics["non_abstain_accuracy"] = total_correct / total if total > 0 else 0
                metrics["non_abstain_total"] = total
                metrics["non_abstain_correct"] = total_correct
                
                metrics["math_accuracy"] = math_correct / math_total if math_total > 0 else 0
                metrics["math_total"] = math_total
                metrics["math_correct"] = math_correct
                
                metrics["choice_accuracy"] = choice_correct / choice_total if choice_total > 0 else 0
                metrics["choice_total"] = choice_total
                metrics["choice_correct"] = choice_correct
                
                metrics["llm_judge_accuracy"] = llm_judge_correct / llm_judge_total if llm_judge_total > 0 else 0
                metrics["llm_judge_total"] = llm_judge_total
                metrics["llm_judge_correct"] = llm_judge_correct
            else:
                metrics["non_abstain_accuracy"] = 0
                metrics["non_abstain_total"] = 0
                metrics["non_abstain_correct"] = 0
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Results for {os.path.basename(src_file)} ({prompt_style}):")
        print(f"{'='*60}")
        print(f"Total samples: {metrics['total']}")
        print(f"Should abstain: {metrics['should_abstain_count']}")
        print(f"Should not abstain: {metrics['should_not_abstain_count']}")
        print(f"\nAbstention Detection:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Abstention Rate: {metrics['abstention_rate']:.4f}")
        
        if not args.skip_correctness:
            print(f"\nCorrectness (Non-abstention samples):")
            print(f"  Overall Accuracy: {metrics.get('non_abstain_accuracy', 0):.4f} ({metrics.get('non_abstain_correct', 0)}/{metrics.get('non_abstain_total', 0)})")
            if metrics.get('math_total', 0) > 0:
                print(f"  Math Accuracy: {metrics.get('math_accuracy', 0):.4f} ({metrics.get('math_correct', 0)}/{metrics.get('math_total', 0)})")
            if metrics.get('choice_total', 0) > 0:
                print(f"  Choice Accuracy: {metrics.get('choice_accuracy', 0):.4f} ({metrics.get('choice_correct', 0)}/{metrics.get('choice_total', 0)})")
            if metrics.get('llm_judge_total', 0) > 0:
                print(f"  LLM Judge Accuracy: {metrics.get('llm_judge_accuracy', 0):.4f} ({metrics.get('llm_judge_correct', 0)}/{metrics.get('llm_judge_total', 0)})")
        
        # Save results
        dataset_name = os.path.basename(src_file).replace('.jsonl', '')
        model_path_parts = args.model_path.rstrip('/').split('/')
        if len(model_path_parts) >= 2:
            model_name = model_path_parts[-2] + '_' + model_path_parts[-1]
        else:
            model_name = model_path_parts[-1] if model_path_parts else 'unknown_model'
        
        output_subdir = os.path.join(
            args.output_dir,
            variant_id if variant_id else "",
            f"{model_name}.{prompt_style}"
        )
        os.makedirs(output_subdir, exist_ok=True)
        
        t = time.localtime()
        timestamp = f"{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}"
        # prefix: user-supplied output_prefix + (optional) variant tag to avoid filename collisions
        _user_prefix = f"{args.output_prefix}_" if args.output_prefix else ""
        _var_prefix = f"{variant_id}_" if variant_id else ""
        prefix = f"{_user_prefix}{_var_prefix}"
        
        # Save detailed results
        detail_file = os.path.join(
            output_subdir, 
            f"{prefix}{dataset_name}.{timestamp}.results.json"
        )
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Save metrics
        metrics_file = os.path.join(
            output_subdir, 
            f"{prefix}{dataset_name}.{timestamp}.metrics.json"
        )
        metrics["dataset"] = dataset_name
        metrics["model"] = model_name
        metrics["prompt_style"] = prompt_style
        metrics["rollout_num"] = args.rollout_num
        metrics["use_llm_judge"] = args.use_llm_judge
        metrics["evaluation_version"] = "v2"
        if variant_id:
            metrics["normal_prompt_variant"] = variant_id
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_subdir}")
        
        all_results[f"{dataset_name}.{prompt_style}" + (f".{variant_id}" if variant_id else "")] = metrics
        
        # ============================================================
        # If this is all_merged, compute and save per-dataset metrics as well
        # ============================================================
        if is_all_merged:
            print(f"\n{'='*60}")
            print("Calculating per-dataset metrics for all_merged...")
            print(f"{'='*60}")
            
            sub_dataset_results = defaultdict(list)
            for result in detailed_results:
                sub_ds = result.get("dataset", "unknown")
                sub_dataset_results[sub_ds].append(result)
            
            per_dataset_metrics = {}
            
            for sub_ds, sub_results in sorted(sub_dataset_results.items()):
                sub_predictions = [{"is_abstention": r["predicted_abstention"]} for r in sub_results]
                sub_gold_labels = [r["should_abstain"] for r in sub_results]
                
                sub_metrics = calculate_abstention_metrics(sub_predictions, sub_gold_labels)
                
                if not args.skip_correctness:
                    sub_non_abstain = [r for r in sub_results 
                                      if not r["should_abstain"] and r.get("reference_answers")]
                    sub_correct = sum(1 for r in sub_non_abstain 
                                     if r.get("correctness_result", {}).get("is_correct", False))
                    sub_total = len(sub_non_abstain)
                    sub_metrics["non_abstain_accuracy"] = sub_correct / sub_total if sub_total > 0 else 0
                    sub_metrics["non_abstain_total"] = sub_total
                    sub_metrics["non_abstain_correct"] = sub_correct
                
                sub_metrics["dataset"] = sub_ds
                sub_metrics["model"] = model_name
                sub_metrics["prompt_style"] = prompt_style
                sub_metrics["rollout_num"] = args.rollout_num
                sub_metrics["use_llm_judge"] = args.use_llm_judge
                sub_metrics["evaluation_version"] = "v2"
                sub_metrics["parent_dataset"] = "all_merged"
                if variant_id:
                    sub_metrics["normal_prompt_variant"] = variant_id
                
                per_dataset_metrics[sub_ds] = sub_metrics
                
                sub_metrics_file = os.path.join(
                    output_subdir,
                    f"{prefix}{sub_ds}.{timestamp}.metrics.json"
                )
                with open(sub_metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(sub_metrics, f, indent=2, ensure_ascii=False)
                
                sub_detail_file = os.path.join(
                    output_subdir,
                    f"{prefix}{sub_ds}.{timestamp}.results.json"
                )
                with open(sub_detail_file, 'w', encoding='utf-8') as f:
                    json.dump(sub_results, f, indent=2, ensure_ascii=False)
                
                print(f"\n{sub_ds} (n={sub_metrics['total']}):")
                print(f"  Abstention: Acc={sub_metrics['accuracy']:.4f}, "
                      f"P={sub_metrics['precision']:.4f}, R={sub_metrics['recall']:.4f}, "
                      f"F1={sub_metrics['f1_score']:.4f}")
                if not args.skip_correctness:
                    print(f"  Correctness: {sub_metrics.get('non_abstain_accuracy', 0):.4f} "
                          f"({sub_metrics.get('non_abstain_correct', 0)}/{sub_metrics.get('non_abstain_total', 0)})")
                
                all_results[f"{sub_ds}.{prompt_style}" + (f".{variant_id}" if variant_id else "")] = sub_metrics
            
            summary_file = os.path.join(
                output_subdir,
                f"{prefix}all_merged_per_dataset.{timestamp}.summary.json"
            )
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(per_dataset_metrics, f, indent=2, ensure_ascii=False)
            
            print(f"\nPer-dataset metrics saved to: {output_subdir}")
    
    # Print summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("Summary of all datasets:")
        print(f"{'='*80}")
        for name, m in all_results.items():
            abstain_acc = m.get('accuracy', 0)
            correct_acc = m.get('non_abstain_accuracy', 0)
            print(f"{name}: Abstain_Acc={abstain_acc:.4f}, Correct_Acc={correct_acc:.4f}, "
                  f"F1={m.get('f1_score', 0):.4f}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
