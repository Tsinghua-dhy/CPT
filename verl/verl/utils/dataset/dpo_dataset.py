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
"""
Offline DPO Dataset for verl.
Each sample contains a prompt, a chosen response, and a rejected response.
The dataset tokenizes them and returns matched pairs for DPO training.
"""

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class DPODataset(Dataset):
    """
    In-memory DPO Dataset that loads (prompt, chosen, rejected) triples.
    Returns tokenized pairs suitable for offline DPO training.
    """

    def __init__(self, parquet_files, tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        chosen_key = config.get("chosen_key", "chosen")
        rejected_key = config.get("rejected_key", "rejected")
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "right")
        use_shm = config.get("use_shm", False)
        self.use_chat_template = config.get("use_chat_template", False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length = max_length

        # Download files
        for i, pf in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(pf, verbose=True, use_shm=use_shm)

        # Read data
        dfs = [pd.read_parquet(pf) for pf in self.parquet_files]
        self.dataframe = pd.concat(dfs, ignore_index=True)
        self.prompts = self.dataframe[prompt_key].tolist()
        self.chosens = self.dataframe[chosen_key].tolist()
        self.rejecteds = self.dataframe[rejected_key].tolist()

    def __len__(self):
        return len(self.prompts)

    def _tokenize_pair(self, prompt_str, response_str):
        """Tokenize a (prompt, response) pair, returning input_ids, attention_mask, loss_mask, and prompt_len."""
        prompt_ids_out = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_out["input_ids"][0]
        prompt_mask = prompt_ids_out["attention_mask"][0]

        response_str_with_eos = response_str + self.tokenizer.eos_token
        resp_ids_out = self.tokenizer(response_str_with_eos, return_tensors="pt", add_special_tokens=False)
        resp_ids = resp_ids_out["input_ids"][0]
        resp_mask = resp_ids_out["attention_mask"][0]

        prompt_len = prompt_ids.shape[0]
        resp_len = resp_ids.shape[0]

        input_ids = torch.cat([prompt_ids, resp_ids], dim=0)
        attention_mask = torch.cat([prompt_mask, resp_mask], dim=0)

        seq_len = input_ids.shape[0]

        # Pad or truncate
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        elif seq_len > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "error":
                raise ValueError(f"Sequence length {seq_len} > max_length {self.max_length}")

        # Loss mask: only on response tokens (exclude prompt and last token)
        loss_mask = attention_mask.clone()
        if prompt_len > 1:
            loss_mask[: min(prompt_len, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_len + resp_len, loss_mask.size(0)) - 1] = 0

        position_ids = compute_position_id_with_mask(attention_mask)

        return input_ids, attention_mask, position_ids, loss_mask

    def __getitem__(self, item):
        prompt = self.prompts[item]
        chosen = self.chosens[item]
        rejected = self.rejecteds[item]

        if self.use_chat_template:
            prompt_chat = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_str = prompt

        c_ids, c_mask, c_pos, c_loss = self._tokenize_pair(prompt_str, chosen)
        r_ids, r_mask, r_pos, r_loss = self._tokenize_pair(prompt_str, rejected)

        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "chosen_position_ids": c_pos,
            "chosen_loss_mask": c_loss,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
            "rejected_position_ids": r_pos,
            "rejected_loss_mask": r_loss,
        }
