# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Optional

import psutil
import torch
from transformers import PreTrainedTokenizer

# ------------------ Reward Configuration ------------------
# read reward config from env vars, fall back to defaults if absent
ANSWER_REWARD = float(os.environ.get("VERL_ANSWER_REWARD", "1.0"))
FORMAT_REWARD = float(os.environ.get("VERL_FORMAT_REWARD", "0.2"))

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(executor, partial(evaluation_func, task, completion, reference, task_extra_info))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        #print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        #print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the
        # exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            # Create tasks for all rows
            tasks_async = [
                single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=300.0)
                for c, r, t, ei in zip(completions, references, tasks, extra_info, strict=True)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            #print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks, strict=True):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result, int | float | bool):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info, num_processes)
        )
    finally:
        loop.close()


def compute_overlong_penalty(response_length: int, max_response_length: int, 
                              buffer_len: int, penalty_factor: float) -> float:
    """
    Compute the overlong penalty: a linear penalty applied in the last
    `buffer_len` tokens of `max_response_length`.

    Example: max_response_length=12288, buffer_len=4096
    - response_length <= 8192:                         penalty = 0
    - response_length = 10240 (mid-buffer):            penalty = -0.5 * penalty_factor
    - response_length >= 12288:                        penalty = -1.0 * penalty_factor

    Args:
        response_length: actual response length
        max_response_length: max generation length
        buffer_len: penalty-buffer length
        penalty_factor: penalty coefficient

    Returns:
        the penalty value (negative or zero).
    """
    # Penalty start position
    penalty_start = max_response_length - buffer_len
    
    if response_length <= penalty_start:
        return 0.0
    
    # Linear penalty: from penalty_start to max_response_length
    # at penalty_start          -> penalty = 0
    # at max_response_length    -> penalty = -penalty_factor
    progress = min(1.0, (response_length - penalty_start) / buffer_len)
    penalty = -progress * penalty_factor
    
    return penalty


# Infer answer_reward and format_reward from a total_reward value.
# Uses values configured via env vars.
# Possible values: 0.0, FORMAT_REWARD, ANSWER_REWARD, ANSWER_REWARD + FORMAT_REWARD
def infer_reward_breakdown(total_reward, answer_reward_value=None, format_reward_value=None):
    # Use env-configured values if no args were passed
    if answer_reward_value is None:
        answer_reward_value = ANSWER_REWARD
    if format_reward_value is None:
        format_reward_value = FORMAT_REWARD
    """
    Infer answer_reward and format_reward from total_reward.

    Possible combinations:
    - 0.0 = wrong answer + invalid format
    - 0.2 = wrong answer + valid format
    - 1.0 = correct answer + invalid format
    - 1.2 = correct answer + valid format
    """
    # Use a tolerance for floating-point comparisons
    eps = 0.01
    
    if abs(total_reward - (answer_reward_value + format_reward_value)) < eps:
        # 1.2 = correct answer + valid format
        return answer_reward_value, format_reward_value
    elif abs(total_reward - answer_reward_value) < eps:
        # 1.0 = correct answer + invalid format
        return answer_reward_value, 0.0
    elif abs(total_reward - format_reward_value) < eps:
        # 0.2 = wrong answer + valid format
        return 0.0, format_reward_value
    else:
        # 0.0 = wrong answer + invalid format
        return 0.0, 0.0


@register("prime")
class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        overlong_buffer_config: Optional[dict] = None,
        max_response_length: int = 8192,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        # Overlong buffer penalty configuration
        self.max_response_length = max_response_length
        if overlong_buffer_config is None:
            overlong_buffer_config = {}
        self.overlong_buffer_enable = overlong_buffer_config.get("enable", False)
        self.overlong_buffer_len = overlong_buffer_config.get("len", 4096)
        self.overlong_buffer_penalty_factor = overlong_buffer_config.get("penalty_factor", 1.0)
        
        if self.overlong_buffer_enable:
            print(f"[prime] Overlong Buffer Penalty: enable={self.overlong_buffer_enable}, "
                  f"max_len={self.max_response_length}, buffer_len={self.overlong_buffer_len}, "
                  f"penalty_factor={self.overlong_buffer_penalty_factor}")

    def verify(self, data):
        """verify the batch and save as ``acc`` tensor"""
        # batched scoring
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)

        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=64,
            )
        except asyncio.TimeoutError:
            #print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            #print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]

        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        # Collect extra information for logging
        ground_truths = []
        response_lengths = []
        answer_rewards = []
        format_rewards = []
        overlong_penalties = []  # overlong penalties per item
        is_overlong_max = []  # whether the response hit max_response_length (used to filter training data later)

        for i in range(len(data)):
            data_source = data_sources[i]
            
            # Get the response length
            resp_len = valid_response_length[i].item()
            response_lengths.append(resp_len)
            
            # Compute the overlong penalty
            if self.overlong_buffer_enable:
                penalty = compute_overlong_penalty(
                    resp_len, 
                    self.max_response_length,
                    self.overlong_buffer_len,
                    self.overlong_buffer_penalty_factor
                )
                # Whether the response hit max_response_length (penalty saturates)
                # When response_length >= max_response_length, penalty == -penalty_factor.
                is_max_overlong = (resp_len >= self.max_response_length)
            else:
                penalty = 0.0
                is_max_overlong = False
            overlong_penalties.append(penalty)
            is_overlong_max.append(is_max_overlong)
            
            # Final reward after applying the penalty
            final_reward = scores[i] + penalty
            reward_tensor[i, resp_len - 1] = final_reward

            # Collect ground_truth
            if "reward_model" in data[i].non_tensor_batch:
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", "")
            else:
                ground_truth = ""
            ground_truths.append(ground_truth)

            # Infer answer_reward and format_reward from total_reward
            ans_r, fmt_r = infer_reward_breakdown(scores[i])
            answer_rewards.append(ans_r)
            format_rewards.append(fmt_r)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if return_dict:
            reward_extra_info = {
                "ground_truth": ground_truths,
                "response_length": response_lengths,
                "total_reward": scores,
                "answer_reward": answer_rewards,
                "format_reward": format_rewards,
                "overlong_penalty": overlong_penalties,
                "is_overlong_max": is_overlong_max,  # whether the response hit max_response_length
                # Overlong buffer config for metrics calculation (scalar values)
                "overlong_buffer_enable": self.overlong_buffer_enable,
                "overlong_buffer_len": self.overlong_buffer_len,
                "max_response_length": self.max_response_length,
            }
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
