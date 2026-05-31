# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import time
import httpx
import os
from openai import OpenAI

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

# ------------------ Reward Configuration ------------------
# read reward config from env vars, fall back to defaults if absent
ANSWER_REWARD = float(os.environ.get("VERL_ANSWER_REWARD", "1.0"))
FORMAT_REWARD = float(os.environ.get("VERL_FORMAT_REWARD", "0.2"))

print(f"[math_verify] Reward Configuration: answer_reward={ANSWER_REWARD}, format_reward={FORMAT_REWARD}")

# ------------------ init GPT client ------------------
# Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) before training if
# you want to enable the LLM-as-a-judge fallback. If unset, math_verify
# silently falls back to rule-based scoring only.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# thread-local storage so each process keeps its own client
import threading
_thread_local = threading.local()

def get_openai_client():
    """get a thread/process-safe OpenAI client"""
    if not OPENAI_API_KEY:
        return None
    if not hasattr(_thread_local, 'client'):
        _thread_local.client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            http_client=httpx.Client(
                base_url=OPENAI_BASE_URL,
                follow_redirects=True,
            ),
        )
    return _thread_local.client

def detect_nonsense_output(text: str, strict_threshold: float = 0.15) -> bool:
    """
    detect whether the text contains large amounts of garbled output
    
    highly simplified: detect only truly garbled multilingual mixtures
    - detect words that mix many scripts (e.g. CJK + Korean + Cyrillic + Hebrew)
    - avoid false positives on normal math reasoning outputs
    
    Args:
        text: text to inspect
        strict_threshold: unused (kept for backward compatibility)
    
    Returns:
        True if garbled output is detected, otherwise False
    """
    if not text or len(text) == 0:
        return False
    
    # count characters from each script across the whole text
    has_cjk = len(re.findall(r'[\u4e00-\u9fff]', text))  # CJK
    has_korean = len(re.findall(r'[\uac00-\ud7af]', text))  # Korean
    has_cyrillic = len(re.findall(r'[\u0400-\u04ff]', text))  # Cyrillic
    has_hebrew = len(re.findall(r'[\u0590-\u05ff]', text))  # Hebrew
    has_thai = len(re.findall(r'[\u0e00-\u0e7f]', text))  # Thai
    
    # count how many scripts are present (a script counts only if it has >= 5 chars)
    language_systems = sum([
        has_cjk >= 5,
        has_korean >= 5,
        has_cyrillic >= 5,
        has_hebrew >= 5,
        has_thai >= 5
    ])
    
    # if 2 or more scripts are mixed, classify as nonsense
    # note: normal math text never mixes CJK / Korean / Cyrillic / Hebrew / Thai
    if language_systems >= 2:
        return True
    
    return False


def extract_boxed_answer(text: str) -> str:
    """
    extract the answer inside the last \\boxed{} in the model output;
    supports nested braces
    """
    # locate the last \boxed occurrence
    last_boxed_idx = text.rfind("\\boxed")
    if last_boxed_idx == -1:
        return ""
    
    # search starting at the \boxed position
    start_text = text[last_boxed_idx:]
    
    # check whether the answer is wrapped in braces
    if "{" not in start_text:
        # no braces; try matching the content directly after \boxed
        pattern = r'\\boxed\s+([^\s\\]+)'
        match = re.search(pattern, start_text)
        if match:
            return match.group(1).strip()
        return ""
    
    # braces are present; handle nesting
    brace_start = start_text.find("{")
    if brace_start == -1:
        return ""
    
    # use a counter to balance nested braces
    brace_count = 0
    brace_end = -1
    
    for i in range(brace_start, len(start_text)):
        if start_text[i] == "{":
            brace_count += 1
        elif start_text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                brace_end = i
                break
    
    if brace_end == -1:
        # closing brace not found
        return ""
    
    # extract content inside the outermost braces
    content = start_text[brace_start + 1:brace_end].strip()
    return content


def gpt4omini_judge(ground_truth: str, pred_answer: str) -> float:
    """
    use GPT-4o-mini as an LLM judge
    returns: 1.0 (correct) or 0.0 (incorrect)
    """
    prompt = f"""You are an expert math evaluator.
Given a gold answer and a predicted answer, judge if they are mathematically consistent.

Ignore formatting (e.g., \\text{{}}, spacing, capitalization).
Accept equivalent expressions (e.g., factored vs expanded form).
Only when the predicted answer is mathematically consistent with the gold answer, output "Correct".

Output format:
Reason: Brief explanation
Judgment: Correct / Incorrect

Input:
Gold: {ground_truth}
Pred: {pred_answer}"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = get_openai_client()  # get the per-process client
            rst = client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                temperature=0.0,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            output = rst.choices[0].message.content.strip()
            
            # parse the judgment
            match = re.search(r"Judgment:\s*(Correct|Partially correct|Incorrect)", output, re.IGNORECASE)
            if match:
                judgment = match.group(1)
                score_map = {
                    "Correct": 1.0,
                    "Incorrect": 0.0
                }
                return score_map.get(judgment, 0.0)
            else:
                # fall back to a permissive substring match
                if "correct" in output.lower() and "incorrect" not in output.lower():
                    return 1.0
                else:
                    return 0.0
                    
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return 0.0
    
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, use_llm_judge: bool = True,
                 allow_multiple_boxed: bool = True, max_chars_after_boxed: int = 1000, 
                 nonsense_penalty: float = -2.0, detect_nonsense: bool = True):
    """
    compute the reward score for a model output
    
    Pipeline:
    1. try to extract content from \\boxed{} in model_output
    2. if a \\boxed{} answer is extracted:
       a. rewrap it as \\boxed{extracted_content}
       b. verify with math_verify
       c. if math_verify passes, return answer_reward + format_reward
       d. if math_verify fails, optionally fall back to LLM-as-a-judge
    3. if no \\boxed{} content was extracted:
       a. return 0.0
    
    Args:
        model_output: raw model output
        ground_truth: gold answer
        timeout_score: fallback score on timeout
        use_llm_judge: enable the LLM-as-a-judge fallback (default True)
        allow_multiple_boxed: whether multiple \\boxed{} are allowed (default True)
        max_chars_after_boxed: max chars allowed after the last \\boxed{} (default 1000)
        nonsense_penalty: penalty when garbled output is detected (unused, kept for compat)
        detect_nonsense: enable garbled-output detection (default True)
    
    Returns:
        float: the computed reward score
            - 1.2 = correct answer + valid format
            - 1.0 = correct answer + invalid format
            - 0.2 = wrong answer + valid format
            - 0.0 = wrong answer + invalid format, or no \boxed{}
    """
    
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    answer_reward_value = 0.0
    format_reward_value = 0.0

    # wrap the ground truth into \boxed{} format
    if "\\boxed{" in ground_truth:
        ground_truth_boxed = ground_truth
    else:
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    
    # Step 1: extract \boxed{} content
    extracted_answer = extract_boxed_answer(model_output)
    
    if extracted_answer:
        # case 1: an answer was extracted
        # wrap it back for math_verify
        pred_boxed = f"\\boxed{{{extracted_answer}}}"
        
        # Step 2a: rule-based math_verify
        try:
            math_verify_result, _ = verify_func([ground_truth_boxed], [pred_boxed])
        except Exception:
            # math_verify raised; treat as fail
            math_verify_result = 0.0
        
        # check format-reward bonus
        if check_format(model_output, allow_multiple_boxed, max_chars_after_boxed):
            format_reward_value = FORMAT_REWARD  # use configured format reward
        
        # Step 2b: math_verify passes => grant answer reward
        if math_verify_result > 0:
            answer_reward_value = ANSWER_REWARD  # use configured answer reward
            ret_score = answer_reward_value + format_reward_value
            return ret_score
        
        # Step 2c: math_verify fails => optionally consult the LLM judge
        if use_llm_judge:
            llm_result = gpt4omini_judge(ground_truth, extracted_answer)
            if llm_result > 0:
                answer_reward_value = ANSWER_REWARD  # use configured answer reward
            else:
                answer_reward_value = 0.0
        else:
            answer_reward_value = timeout_score
        
        ret_score = answer_reward_value + format_reward_value
        return ret_score
    
    # no \boxed{} content extracted => zero score
    return 0.0


def check_format(solution_str, allow_multiple_boxed: bool = False, max_chars_after_boxed: int = 1000) -> bool:
    """
    check whether the answer follows the required format
    
    Args:
        solution_str: string to check
        allow_multiple_boxed: whether multiple \\boxed{} are allowed (default False)
        max_chars_after_boxed: max chars allowed after the last \\boxed{} (default 1000)
    
    Returns:
        whether the format is valid
    """
    # 1. must contain a \boxed{} match
    if "\\boxed{" not in solution_str and "\\boxed " not in solution_str:
        return False
    
    # 2. characters after \boxed{} must not exceed the limit (if set)
    if max_chars_after_boxed is not None:
        idx = solution_str.rfind("\\boxed")
        if idx < 0:
            idx = solution_str.rfind("\\fbox")
        if idx >= 0:
            after = solution_str[idx:]
            if len(after) > max_chars_after_boxed:
                return False
    
    # 3. \boxed{} must be unique (unless multiple allowed)
    if not allow_multiple_boxed:
        count = solution_str.count("\\boxed{") + solution_str.count("\\boxed ")
        if count != 1:
            return False
    
    if detect_nonsense_output(solution_str):
        return False

    return True
