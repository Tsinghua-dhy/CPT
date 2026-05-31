import argparse
import torch.distributed as dist
import json
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from openai import OpenAI
import sys
import os
import re
from datasets import load_dataset
import json
import copy
from tqdm import tqdm
import requests
from collections import defaultdict
import time
from time import sleep
import multiprocessing
import torch
import random
from evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=0)
    # Now supports multiple input files
    parser.add_argument("--src_files", type=str, nargs='+', required=True,
                        help="List of input .jsonl files to evaluate")
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3")
    parser.add_argument("--model_path", type=str, default="None")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--port", type=str, default="5004")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=3072)
    return parser.parse_args()
def process_text(example, model_short_name, tokenizer):
    # Fixed prompt for base models
    question_raw = example["question"]
    prompt = f"""The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The final answer should be enclosed within \\boxed{{}}.

User:{question_raw}
Assistant:"""
    example["chat_prompt"] = prompt
    return example

def process_text_wo_retrieve(example, model_short_name, tokenizer):
    # Fixed prompt for base models
    question = example["question"]
    user_prompt = (
        'Please answer the following math question. You should think step by step to solve it.\n\n'
        'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
        f'Question:\n{question}\n\n')
    messages_chat = [
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True)
    example["chat_prompt"] = prompt
    return example

def extract_answer_math(s):
    """
    Extract the answer from a math model's output.
    Supports several formats: \boxed{}, <answer>...</answer>, "the correct answer is"
    """
    extracted_text = ''
    
    # Method 1: balanced-brace matching for \boxed{}
    def extract_boxed_content(text):
        """Extract \boxed{} content, handling nested braces correctly."""
        pattern = r'\\boxed\{'
        matches = []
        
        for match in re.finditer(pattern, text):
            start_pos = match.end() - 1  # locate the opening '{'
            brace_count = 0
            content_start = start_pos + 1
            
            for i in range(start_pos, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the matching closing brace
                        matches.append(text[content_start:i])
                        break
        
        return matches
    
    # All matches
    all_matches = []
    
    # 1. Extract \boxed{} content (balanced-brace matching) 
    boxed_matches = extract_boxed_content(s)
    all_matches.extend(boxed_matches)
    
    # 2. Extract <answer>...</answer> content
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, s, re.DOTALL | re.IGNORECASE)
    all_matches.extend(answer_matches)
    
    # 3. Extract the option after "the correct answer is"
    choice_pattern = r'[Tt]he correct answer is:?\s*([A-Ja-j])'
    choice_matches = re.findall(choice_pattern, s, re.IGNORECASE)
    all_matches.extend(choice_matches)
    
    # Return the last match (usually the final answer) 
    if all_matches:
        extracted_text = all_matches[-1]
    
    return extracted_text.strip()

def main():
    print("=Begin="*10)
    args = parse_args()
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    temp=args.temp
    port=args.port
    t_start = time.time()
    model_path=args.model_path
    # Derive model_short_name from the second-to-last path component
    model_short_name = model_path.split('/')[-2].lower() + model_path.split('/')[-1].lower()  # e.g., 'qwen-3-4b-base-math-grpo'
    gpu_memory_rate=args.gpu_memory_rate
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_gpus = torch.cuda.device_count()
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=gpu_memory_rate, trust_remote_code=True)
    # Iterate over each src_file
    for src_file in args.src_files:
        print(f"\n{'='*80}")
        print(f"Processing file: {src_file}")
        print(f"{'='*80}")

        # Read the current file
        data_ori_all = []
        with open(src_file, "r") as f:
            for i, line in enumerate(f):
                if args.start_sample <= i < args.end_sample:
                    obj_ori = json.loads(line)
                    data_ori_all.append(obj_ori)
                if i >= args.end_sample - 1:
                    break

        print("All Data Length:", len(data_ori_all))

        print("All Data Length: ",len(data_ori_all))
        chunk_size = 20000
        chunk_num = len(data_ori_all) // chunk_size
        if len(data_ori_all) % chunk_size != 0:
            chunk_num += 1

        for h in range(chunk_num):
            print("==" * 80)
            print("Begin Chunk: ", h, "All: ", chunk_num)
            data_ori = data_ori_all[h * chunk_size:(h + 1) * chunk_size]
            data = [copy.deepcopy(item) for item in data_ori]
            # process prompt
            for item in data:
                if model_short_name == "qwen-2.5-7b-instruct-ur2-mathkqa":
                    item = process_text_wo_retrieve(item, model_short_name, tokenizer)
                else:
                    item = process_text(item, model_short_name, tokenizer)
                item["gen_text_store"] = ""

            finished_all_list = []
            # Trim stop_tokens
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]
            sampling_params = SamplingParams(temperature=temp, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens, stop=stop_tokens)

            # Run a single generation pass
            prompts = [item["chat_prompt"] for item in data]
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                quesiton = data[i]["question"]
                answer = data[i]["answer"]
                generated_text = output.outputs[0].text
                gen_text_store = generated_text.strip()

                # extractanswer
                pred_ans = extract_answer_math(generated_text)
                if pred_ans == '':
                    pred_ans = "I don't know."

                original_data = {
                    "question": quesiton,
                    "answer": answer,
                    "pred_ans": pred_ans,
                    "stop_reason_final": "finished",
                    "gen_text_store": gen_text_store,
                }
                finished_all_list.append(original_data)

            print("=="*80)
            print("Chunk Finished: ", h, "Samples: ", len(finished_all_list))

        input_list = []
        output_list = []

        for item in finished_all_list:
            if model_short_name == "qwen-2.5-7b-instruct-ur2-mathkqa":
                processed_item = process_text_wo_retrieve(item, model_short_name, tokenizer)
                input_prompt = processed_item['chat_prompt']
            else:
                processed_item = process_text(item, model_short_name, tokenizer)
                input_prompt = processed_item['chat_prompt']
            input_list.append(input_prompt)
            output_list.append(item['gen_text_store'])
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
        os.makedirs(output_dir, exist_ok=True)
        split = "test"
        run_evaluation(        
            finished_all_list,
            input_list,
            output_list,
            output_dir,
            time.time()-t_start,
            split,
        )
        if dist.is_initialized():
                dist.destroy_process_group()
if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()