"""
Use the SFT'd 32B base judge model to score pairwise reasoning paths. 

Model: olmo3-32b-mathjudger-v2-sft (SFT'd from a base model).
- chat template cannot be used
- prompts must follow the raw training format exactly (User: ... Assistant:) 
- the model writes its judgment inside \boxed{}
- the Confidence section has been removed (in practice the model only emits 'high' / 'very high', which is uninformative) 

outputeach pair: 
- all_judgments: judgments from each of `num_rollouts` rollouts
- consensus / final_judgment / consensus_count
- judgment_difficulty: simple / medium / hard / no_consensus (based purely on consensus strength) 
"""
import json
import re
import traceback
import torch
from pathlib import Path
from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ========== User-configurable settings ==========
model_path = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # large model path
input_file = "judged_pairs/pairs_data_split_3.jsonl"  # inputfile
output_file = "judged_pairs/pairs_data_split_3_judged.jsonl"  # outputfile
gpu_memory_utilization = 0.9
tensor_parallel_size = 8
max_generation_tokens = 4096  # max generation tokens
num_rollouts = 8              # number of judgements per pair
consensus_threshold = 5       # minimum agreements for consensus (5/8) 
temperature = 0.7             # sampling temperature
top_p = 0.95
top_k = 20
stop_tokens = ["<|im_end|>", "<|endoftext|>"]
# =================================

# ========== Prompt template (must exactly match the SFT training format) ==========
# Training format: User: ... Assistant:
# Note: the leading "role description" + "task description" is the User content
PROMPT_TEMPLATE = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The final answer should be enclosed within \boxed{{}}.
User: You are an expert mathematical reasoning evaluator. You will be given:
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
{reasoning_b}
Assistant:"""


def extract_boxed_content(text):
    """Extract every \\boxed{...} body from `text` in order of appearance."""
    pattern = r'\\boxed\{'
    matches = []
    for m in re.finditer(pattern, text):
        start_pos = m.end() - 1  # points at '{'
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


def normalize_judgment(raw):
    """Normalize the model output's judgment string to the canonical 4-way label."""
    if raw is None:
        return ""
    s = raw.strip().lower()
    if 'path a is better' in s or ('path a' in s and 'better' in s):
        return "Path A is better"
    if 'path b is better' in s or ('path b' in s and 'better' in s):
        return "Path B is better"
    if 'equally good' in s or 'both are good' in s:
        return "Both are equally good"
    if 'equally bad' in s or 'both are bad' in s:
        return "Both are equally bad"
    return ""  # no match -> invalid


def parse_judge_output(text):
    """
    parsemodel output, return (analysis, judgment). 
    Take the judgment from the last \\boxed{}; if absent, fall back to a regex scan of the whole text. 
    analysis = the text before \\boxed{} (with leading/trailing whitespace stripped). 
    """
    text = (text or "").strip()
    if not text:
        return "", ""

    boxed = extract_boxed_content(text)
    judgment = normalize_judgment(boxed[-1]) if boxed else ""
    if not judgment:
        # fallback: scan the whole text for one of the 4 labels
        judgment = normalize_judgment(text)

    # analysis takes the content BEFORE \boxed{}
    if boxed:
        last_box_pos = text.rfind('\\boxed{')
        analysis = text[:last_box_pos].strip() if last_box_pos > 0 else text
    else:
        analysis = text

    return analysis, judgment


def build_judge_prompt(question, ground_truth, answer_a, reasoning_a, answer_b, reasoning_b):
    """Build the judge prompt (base model, no chat template)."""
    return PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        answer_a=answer_a,
        reasoning_a=reasoning_a,
        answer_b=answer_b,
        reasoning_b=reasoning_b,
    )


def extract_paths_from_pair(pair_data):
    """Extract the two paths from a pair item."""
    question = pair_data.get("question", "")
    ground_truth = pair_data.get("ground_truth", "")

    path_a = pair_data.get("path_a", {}) or {}
    path_b = pair_data.get("path_b", {}) or {}

    answer_a = path_a.get("pred_ans", "")
    reasoning_a = path_a.get("gen_text", "")
    answer_b = path_b.get("pred_ans", "")
    reasoning_b = path_b.get("gen_text", "")

    return question, ground_truth, answer_a, reasoning_a, answer_b, reasoning_b


def determine_consensus(judgments):
    """Check whether the rollouts reach consensus. Empty strings are treated as invalid."""
    valid = [j for j in judgments if j]
    if not valid:
        return False, None, 0

    counter = Counter(valid)
    judgment, count = counter.most_common(1)[0]
    if count >= consensus_threshold:
        return True, judgment, count
    return False, None, 0


def calculate_judgment_difficulty(judge_results):
    """
    Bucket judgement difficulty purely by consensus strength (no confidence axis). 

    - no_consensus: consensus_count < threshold
    - simple : 8/8 unanimous
    - medium : 6 or 7 agreeing rollouts
    - hard   : exactly 5 agreeing rollouts (just at the threshold), or >=3 distinct judgments
    """
    consensus_count = judge_results['consensus_count']
    has_consensus = judge_results['has_consensus']
    all_judgments = judge_results['all_judgments']
    total_rollouts = judge_results['total_rollouts']

    valid_judgments = [j for j in all_judgments if j]
    unique_judgments = len(set(valid_judgments))
    valid_count = len(valid_judgments)
    consensus_rate = consensus_count / total_rollouts if total_rollouts > 0 else 0

    metrics = {
        "consensus_rate": round(consensus_rate, 3),
        "valid_rate": round(valid_count / total_rollouts, 3) if total_rollouts > 0 else 0,
        "unique_judgments_count": unique_judgments,
    }

    if not has_consensus or consensus_count < consensus_threshold:
        return "no_consensus", metrics

    if consensus_count >= 8:
        return "simple", metrics
    if consensus_count >= 6:
        return "medium", metrics
    # consensus_count == 5
    if unique_judgments >= 3:
        return "hard", metrics
    return "hard", metrics


def main():
    # Load the tokenizer (only used inside vLLM; prompts skip the chat template)
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _ = tokenizer  # keep the reference to silence linters

    # Load the teacher model (matches eval_math500_cot_new.py: tp_size + gmu + trust_remote_code only)
    print("Loading LLM ...")
    if tensor_parallel_size <= 0:
        tp_size = max(1, torch.cuda.device_count())
    else:
        tp_size = tensor_parallel_size
    print(f"  tensor_parallel_size = {tp_size}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    # readinputdata
    print(f"Loading input data from {input_file} ...")
    pairs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"Total pairs to judge: {len(pairs)}")

    # Make sure the output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ============ Build all prompts in a batch ============
    print("Building all prompts ...")
    all_prompts = []
    pair_indices = []  # each prompt -> (pair_idx, rollout_idx)
    valid_pairs = []   # only keep valid (pair_idx, pair_data) tuples

    for pair_idx, pair_data in enumerate(pairs):
        result = extract_paths_from_pair(pair_data)
        if result is None:
            print(f"Skipping pair {pair_idx + 1}: invalid data structure")
            continue
        question, ground_truth, answer_a, reasoning_a, answer_b, reasoning_b = result

        prompt = build_judge_prompt(
            question, ground_truth,
            answer_a, reasoning_a,
            answer_b, reasoning_b,
        )

        for rollout_idx in range(num_rollouts):
            all_prompts.append(prompt)
            pair_indices.append((pair_idx, rollout_idx))

        valid_pairs.append((pair_idx, pair_data))

    print(f"Total prompts to process: {len(all_prompts)} "
          f"({len(valid_pairs)} pairs × {num_rollouts} rollouts)")

    # ============ Single batched inference call ============
    print("Running batch inference ...")
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_generation_tokens,
        top_p=top_p,
        top_k=top_k,
        stop=stop_tokens,
    )
    outputs = llm.generate(all_prompts, sampling_params)
    print(f"Batch inference completed! Got {len(outputs)} outputs")

    # ============ Parse all outputs and aggregate at the pair level ============
    print("Parsing outputs and organizing results ...")
    pair_results = {pair_idx: {'all_judgments': [], 'all_analyses': []}
                    for pair_idx, _ in valid_pairs}

    for output, (pair_idx, _rollout_idx) in zip(outputs, pair_indices):
        generated_text = output.outputs[0].text
        analysis, judgment = parse_judge_output(generated_text)
        pair_results[pair_idx]['all_analyses'].append(analysis)
        pair_results[pair_idx]['all_judgments'].append(judgment)

    # ============ Write the final result for each pair ============
    print("Processing consensus and writing results ...")
    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for pair_idx, pair_data in valid_pairs:
            try:
                all_judgments = pair_results[pair_idx]['all_judgments']
                all_analyses = pair_results[pair_idx]['all_analyses']

                has_consensus, final_judgment, consensus_count = determine_consensus(all_judgments)

                judge_results = {
                    "all_judgments": all_judgments,
                    "all_analyses": all_analyses,
                    "has_consensus": has_consensus,
                    "final_judgment": final_judgment if has_consensus else None,
                    "consensus_count": consensus_count,
                    "total_rollouts": num_rollouts,
                }

                judgment_difficulty, judgment_metrics = calculate_judgment_difficulty(judge_results)
                judge_results["judgment_difficulty"] = judgment_difficulty
                judge_results["judgment_metrics"] = judgment_metrics

                judged_pair = pair_data.copy()
                judged_pair["judge_results"] = judge_results
                out_f.write(json.dumps(judged_pair, ensure_ascii=False) + '\n')

                processed_count += 1
                if processed_count % 100 == 0 or processed_count == len(valid_pairs):
                    print(f"Processed: {processed_count}/{len(valid_pairs)}  "
                          f"last: consensus={has_consensus}, judgment={final_judgment}, "
                          f"difficulty={judgment_difficulty}")
            except Exception as e:
                print(f"Error processing pair {pair_idx + 1}: {e}")
                traceback.print_exc()
                continue

        out_f.flush()

    print(f"\n=== Processing Complete ===")
    print(f"Total processed: {processed_count}/{len(pairs)}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
