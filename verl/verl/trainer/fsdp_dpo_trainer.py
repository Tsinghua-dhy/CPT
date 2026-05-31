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
A lightweight offline DPO Trainer built on top of the FSDP SFT Trainer.
Reuses model building, optimizer, checkpointing, and distributed infra from SFT.
Only overrides loss computation and dataset creation.
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig, AutoModelForCausalLM

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, extract_step
from verl.utils.dataset.dpo_dataset import DPODataset
from verl.utils.device import get_device_id, get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.torch_dtypes import PrecisionType

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_DPO_LOGGING_LEVEL", "WARN"))


def compute_log_probs(logits, labels, loss_mask):
    """
    Compute per-sequence sum of log probabilities.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) — input_ids shifted by the caller
        loss_mask: (batch, seq_len) — 1 on response tokens, 0 elsewhere

    Returns:
        (batch,) sum of log probs over response tokens
    """
    # Shift: logits[..., :-1, :] predicts labels[..., 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, :-1].contiguous().float()

    per_token_logps = -F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_logits.size(0), shift_logits.size(1))

    return (per_token_logps * shift_mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
    label_smoothing=0.0,
    loss_type="sigmoid",
    reference_free=False,
):
    """Standard offline DPO loss."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    if reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)

    logits = pi_logratios - ref_logratios

    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "ipo":
        losses = (logits - 1.0 / (2.0 * beta)) ** 2
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


class FSDPDPOTrainer(FSDPSFTTrainer):
    """
    Offline DPO trainer. Inherits all infra from FSDPSFTTrainer.
    Overrides:
      - _compute_loss_and_backward  →  DPO loss with ref model
      - training_step               →  returns DPO-specific metrics
      - validation_step             →  DPO loss on val set
    After model build, creates a frozen copy as the reference model.
    """

    def __init__(self, config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset):
        # Store DPO config before super().__init__ (which calls _build_model_optimizer)
        self.dpo_config = config.get("dpo", {})
        self.beta = self.dpo_config.get("beta", 0.1)
        self.loss_type = self.dpo_config.get("loss_type", "sigmoid")
        self.label_smoothing = self.dpo_config.get("label_smoothing", 0.0)
        self.reference_free = self.dpo_config.get("reference_free", False)

        super().__init__(config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset)

        if device_mesh.get_rank() == 0:
            print(f"[DPO] beta={self.beta}, loss_type={self.loss_type}, "
                  f"label_smoothing={self.label_smoothing}, reference_free={self.reference_free}")

    def _build_model_optimizer(self):
        """Build model + optimizer (from SFT), then create frozen ref model by loading from the same pretrained path."""
        super()._build_model_optimizer()

        if not self.reference_free:
            if self.device_mesh.get_rank() == 0:
                print("[DPO] Building reference model from pretrained weights...")

            log_gpu_memory_usage("[DPO] Before ref model allocation", logger=logger)

            local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=False)
            trust_remote_code = self.config.model.trust_remote_code
            torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

            model_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
            if hasattr(model_config, "max_position_embeddings"):
                model_config.max_position_embeddings = max(
                    model_config.max_position_embeddings, self.config.data.max_length
                )

            init_context = get_init_weight_context_manager(
                use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
            )

            with init_context():
                ref_model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    config=model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                )

            # Freeze all parameters
            for p in ref_model.parameters():
                p.requires_grad = False

            # Wrap with FSDP (same config as policy model)
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            )
            auto_wrap_policy = get_fsdp_wrap_policy(
                ref_model,
                config=self.config.model.fsdp_config.wrap_policy,
                is_lora=False,
            )

            self.ref_model = FSDP(
                ref_model,
                cpu_offload=None,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
            self.ref_model.eval()

            log_gpu_memory_usage("[DPO] After ref model allocation", logger=logger)

            if self.device_mesh.get_rank() == 0:
                print("[DPO] Reference model ready (frozen, FSDP-wrapped).")
        else:
            self.ref_model = None

    def _concatenated_forward(self, model, batch):
        """Run a SINGLE forward pass on concatenated chosen+rejected inputs.

        FSDP FULL_SHARD requires exactly one forward per backward on the same
        model to keep its internal all-gather / reduce-scatter hooks in order.
        Doing two separate forwards on the same FSDP model before one backward
        corrupts the hook execution order and causes cross-node hangs.

        Returns:
            chosen_logps: (batch_size,)
            rejected_logps: (batch_size,)
        """
        device = self.device_name
        c_ids = batch["chosen_input_ids"].to(device)
        c_mask = batch["chosen_attention_mask"].to(device)
        c_pos = batch["chosen_position_ids"].to(device)
        c_loss_mask = batch["chosen_loss_mask"].to(device)

        r_ids = batch["rejected_input_ids"].to(device)
        r_mask = batch["rejected_attention_mask"].to(device)
        r_pos = batch["rejected_position_ids"].to(device)
        r_loss_mask = batch["rejected_loss_mask"].to(device)

        bsz = c_ids.size(0)

        # Concatenate chosen and rejected into one batch: [chosen; rejected]
        cat_ids = torch.cat([c_ids, r_ids], dim=0)
        cat_mask = torch.cat([c_mask, r_mask], dim=0)
        cat_pos = torch.cat([c_pos, r_pos], dim=0)
        cat_loss_mask = torch.cat([c_loss_mask, r_loss_mask], dim=0)

        # Single forward pass
        output = model(
            input_ids=cat_ids,
            attention_mask=cat_mask,
            position_ids=cat_pos,
            use_cache=False,
        )

        # Compute log probs on the concatenated output, then split
        all_logps = compute_log_probs(output.logits, cat_ids, cat_loss_mask)
        chosen_logps = all_logps[:bsz]
        rejected_logps = all_logps[bsz:]

        return chosen_logps, rejected_logps

    def _compute_dpo_loss(self, batch, do_backward=True, loss_scale=1.0):
        """Compute DPO loss on a batch of (chosen, rejected) pairs.

        Args:
            batch: TensorDict with chosen/rejected input_ids, attention_mask, etc.
            do_backward: if True, call loss.backward() inside (required for FSDP correctness).
            loss_scale: scale factor applied before backward (for gradient accumulation).
        """
        device = self.device_name

        # ---- Reference log probs (compute FIRST, single forward, then free) ----
        if self.ref_model is not None:
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                ref_chosen_logps, ref_rejected_logps = self._concatenated_forward(self.ref_model, batch)
                ref_chosen_logps = ref_chosen_logps.detach()
                ref_rejected_logps = ref_rejected_logps.detach()
        else:
            bsz = batch["chosen_input_ids"].size(0)
            ref_chosen_logps = torch.zeros(bsz, device=device)
            ref_rejected_logps = torch.zeros(bsz, device=device)

        # ---- Policy log probs (single forward, need gradients) ----
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            policy_chosen_logps, policy_rejected_logps = self._concatenated_forward(self.fsdp_model, batch)

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                beta=self.beta,
                label_smoothing=self.label_smoothing,
                loss_type=self.loss_type,
                reference_free=self.reference_free,
            )

        if do_backward:
            (loss * loss_scale).backward()

        # Accuracy: how often policy prefers chosen over rejected
        with torch.no_grad():
            accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean()

        return loss, chosen_rewards, rejected_rewards, accuracy

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()
        self.optimizer.zero_grad()

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)

        total_loss = 0.0
        total_chosen_rewards = 0.0
        total_rejected_rewards = 0.0
        total_accuracy = 0.0

        for mb in micro_batches:
            # backward inside _compute_dpo_loss to keep FSDP forward/backward in sync
            loss, c_rew, r_rew, acc = self._compute_dpo_loss(
                mb, do_backward=True, loss_scale=1.0 / n_micro_batches
            )
            total_loss += loss.item() / n_micro_batches
            total_chosen_rewards += c_rew.item() / n_micro_batches
            total_rejected_rewards += r_rew.item() / n_micro_batches
            total_accuracy += acc.item() / n_micro_batches

        # Grad clip
        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]

        # All-reduce metrics
        metrics_tensor = torch.tensor(
            [total_loss, total_chosen_rewards, total_rejected_rewards, total_accuracy],
            device=self.device_name,
        )
        torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.AVG)

        return {
            "train/dpo_loss": metrics_tensor[0].item(),
            "train/chosen_rewards": metrics_tensor[1].item(),
            "train/rejected_rewards": metrics_tensor[2].item(),
            "train/reward_margin": (metrics_tensor[1] - metrics_tensor[2]).item(),
            "train/accuracy": metrics_tensor[3].item(),
            "train/lr(1e-3)": lr * 1e3,
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss, c_rew, r_rew, acc = self._compute_dpo_loss(batch, do_backward=False, loss_scale=1.0)
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss


def run_dpo(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    from verl.utils import hf_tokenizer
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    train_dataset = DPODataset(parquet_files=config.data.train_files, tokenizer=tokenizer, config=config.data)
    val_dataset = DPODataset(parquet_files=config.data.val_files, tokenizer=tokenizer, config=config.data)

    trainer = FSDPDPOTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
def main(config):
    run_dpo(config)


if __name__ == "__main__":
    main()
