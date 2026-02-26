# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty,
    compute_memory_efficient_kl,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None, tokenizer=None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()
        
        # OPD special token masking
        self._opd_first_mask_token_ids = {}  
        if self.tokenizer is not None and self.config.get("opd_mask_special_tokens", False):
            self._init_opd_mask_tokens()

    def _init_opd_mask_tokens(self):
        """Initialize token IDs for OPD special token masking.
        """
        if self.tokenizer is None:
            return
        
        # Tokens to mask on first occurrence only
        # Default: ["<", "think"]
        opd_first_occurrence_tokens = self.config.get("opd_mask_first_tokens", ["<", "think", "<|im_end|>"]) # we directly replace <endoftext> with <|im_end|>
        
        # Get token IDs for each token (first occurrence masking)
        self._opd_first_mask_token_ids = {}  # token_id -> True (to track first occurrence)
        for token in opd_first_occurrence_tokens:
            try:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                # For single-token cases, store the token ID
                if len(ids) == 1:
                    self._opd_first_mask_token_ids[ids[0]] = token
                else:
                    # For multi-token cases (like <|im_end|> might be), store all IDs
                    for tid in ids:
                        self._opd_first_mask_token_ids[tid] = token
            except Exception:
                pass  # Token may not exist in vocab
        
        if self._opd_first_mask_token_ids:
            print(f"[OPD Masking] Initialized first-occurrence masking for token_ids: {self._opd_first_mask_token_ids}")

    def _compute_opd_kl_mask(self, responses: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Compute mask for OPD KL loss, masking special tokens that cause vocab mismatch.
        
        Masks the FIRST occurrence of each specified token (e.g., "<", "think", "<|im_end|>").
        
        Args:
            responses: (batch_size, response_length) - token IDs of responses
            response_mask: (batch_size, response_length) - original response mask
            
        Returns:
            kl_mask: (batch_size, response_length) - mask with special tokens zeroed out
        """
        if not self.config.get("opd_mask_special_tokens", False):
            return response_mask
        
        if not self._opd_first_mask_token_ids:
            return response_mask
            
        kl_mask = response_mask.clone()
        batch_size, response_length = responses.shape

        # total_masked = 0
        
        # For each token ID that needs first-occurrence masking
        for token_id in self._opd_first_mask_token_ids.keys():
            # Find first occurrence in each batch and mask it
            for b in range(batch_size):
                # Find positions where this token appears
                matches = (responses[b] == token_id).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    # Mask only the first occurrence
                    first_pos = matches[0].item()
                    kl_mask[b, first_pos] = 0
        #             total_masked += 1
        #             print(f"[OPD Masking] Masked token {token_id} at position {first_pos} in batch {b}")

        # print(f"[OPD DEBUG] Total tokens masked this batch: {total_masked}")
        # print(f"[OPD DEBUG] response_mask sum: {response_mask.sum().item()}, kl_mask sum: {kl_mask.sum().item()}")
        return kl_mask

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # Debug: find which param has mismatched grad
        # print(f"[DEBUG OPTIMIZER] Checking param/grad dtypes and devices...")
        # for i, (name, p) in enumerate(self.actor_module.named_parameters()):
        #     if p.grad is not None:
        #         if p.dtype != p.grad.dtype or p.device != p.grad.device:
        #             print(f"  MISMATCH param[{i}] {name}: "
        #                   f"param(device={p.device}, dtype={p.dtype}, shape={p.shape}) vs "
        #                   f"grad(device={p.grad.device}, dtype={p.grad.dtype}, shape={p.grad.shape})")

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    # =========================================================================
    # EMA-PG STYLE: LOGITS-BASED KL METHODS
    # =========================================================================

    def _forward_micro_batch_with_logits(
        self, micro_batch, temperature,
        kl_topk_k: int = None, kl_topk_indices: torch.Tensor = None,
    ) -> tuple:
        """
        copied from https://github.com/LunjunZhang/ema-pg
        Forward pass that also returns logits for KL divergence computation.

        Args:
            micro_batch: Input batch dict.
            temperature: Temperature for logits.
            kl_topk_k: -1 for full logits, >0 to compute top-k from own logits.
                       None means use kl_topk_indices instead.
            kl_topk_indices: Indices to gather at (only used when kl_topk_k is None).

        Returns:
            entropy: (bs, response_len)
            log_probs: (bs, response_len)
            kl_inputs: dict containing:
                - logits_k: (bs, response_len, vocab_size) if kl_topk_k=-1, else (bs, response_len, k)
                - topk_indices: (bs, response_len, k) or None if kl_topk_k=-1
                - logsumexp: (bs, response_len) - for proper normalization
        """
        if kl_topk_k is None and kl_topk_indices is None:
            raise ValueError("Must provide either kl_topk_k or kl_topk_indices")
        if kl_topk_k is not None and kl_topk_indices is not None:
            raise ValueError("kl_topk_k and kl_topk_indices are mutually exclusive")

        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature)

                # Compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                # Compute log probs
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # Derive logsumexp from log_probs: logsumexp = logits[label] - log_softmax(logits)[label]
                logits_at_label_rmpad = logits_rmpad.gather(-1, input_ids_rmpad_rolled.unsqueeze(-1)).squeeze(-1)
                logsumexp_rmpad = logits_at_label_rmpad - log_probs

                # Handle top-k or full logits
                if kl_topk_k == -1:
                    logits_k_rmpad = logits_rmpad
                    topk_indices_rmpad = None
                elif kl_topk_k is not None and kl_topk_k > 0:
                    _, topk_indices_rmpad = logits_rmpad.topk(kl_topk_k, dim=-1)
                    logits_k_rmpad = logits_rmpad.gather(-1, topk_indices_rmpad)
                else:
                    # kl_topk_k is None, use kl_topk_indices - gather later
                    logits_k_rmpad = None
                    topk_indices_rmpad = None

                # Gather if sp > 1
                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if logits_k_rmpad is not None:
                        logits_k_rmpad = gather_outpus_and_unpad(logits_k_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    else:
                        logits_rmpad = gather_outpus_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    logsumexp_rmpad = gather_outpus_and_unpad(logsumexp_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if topk_indices_rmpad is not None:
                        topk_indices_rmpad = gather_outpus_and_unpad(topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # Handle kl_topk_indices case: gather directly at response positions
                if kl_topk_k is None and kl_topk_indices is not None:
                    # Validate kl_topk_indices shape
                    assert kl_topk_indices.shape[0] == batch_size, \
                        f"kl_topk_indices batch size {kl_topk_indices.shape[0]} != expected {batch_size}"
                    assert kl_topk_indices.shape[1] == response_length, \
                        f"kl_topk_indices response_length {kl_topk_indices.shape[1]} != expected {response_length}"

                    total_nnz = logsumexp_rmpad.shape[0]
                    inverse_indices = torch.full((batch_size * seqlen,), -1, dtype=torch.long, device=logsumexp_rmpad.device)
                    inverse_indices[indices] = torch.arange(total_nnz, device=logsumexp_rmpad.device)

                    response_start = seqlen - response_length - 1
                    batch_offsets = torch.arange(batch_size, device=logsumexp_rmpad.device) * seqlen
                    response_offsets = torch.arange(response_length, device=logsumexp_rmpad.device)
                    flattened_response_pos = batch_offsets.unsqueeze(1) + response_start + response_offsets.unsqueeze(0)
                    rmpad_response_pos = inverse_indices[flattened_response_pos]
                    valid_mask = rmpad_response_pos >= 0
                    safe_rmpad_pos = rmpad_response_pos.clamp(min=0)

                    k = kl_topk_indices.shape[-1]
                    rmpad_pos_expanded = safe_rmpad_pos.unsqueeze(-1).expand(-1, -1, k)
                    logits_k = logits_rmpad[rmpad_pos_expanded.reshape(-1), kl_topk_indices.reshape(-1)]
                    logits_k = logits_k.reshape(batch_size, response_length, k)
                    logsumexp = logsumexp_rmpad[safe_rmpad_pos]
                    logits_k = logits_k * valid_mask.unsqueeze(-1)
                    logsumexp = logsumexp * valid_mask

                    kl_inputs = {"logits_k": logits_k, "topk_indices": kl_topk_indices, "logsumexp": logsumexp}
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                    full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]
                else:
                    # Standard path: pad back to (bsz, seqlen), then slice
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_logits_k = pad_input(hidden_states=logits_k_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                    full_logsumexp = pad_input(hidden_states=logsumexp_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    if topk_indices_rmpad is not None:
                        full_topk_indices = pad_input(hidden_states=topk_indices_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)

                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]
                    logits_k = full_logits_k[:, -response_length - 1:-1, :]
                    logsumexp = full_logsumexp.squeeze(-1)[:, -response_length - 1:-1]

                    if kl_topk_k == -1:
                        kl_inputs = {"logits_k": logits_k, "topk_indices": None, "logsumexp": logsumexp}
                    else:
                        topk_indices = full_topk_indices[:, -response_length - 1:-1, :]
                        kl_inputs = {"logits_k": logits_k, "topk_indices": topk_indices, "logsumexp": logsumexp}

                # Shape assertions for rmpad branch
                batch_size_out = log_probs.shape[0]
                seq_len_out = log_probs.shape[1]
                assert kl_inputs["logsumexp"].shape == (batch_size_out, seq_len_out), \
                    f"logsumexp shape {kl_inputs['logsumexp'].shape} != expected ({batch_size_out}, {seq_len_out})"
                assert kl_inputs["logits_k"].shape[0] == batch_size_out and kl_inputs["logits_k"].shape[1] == seq_len_out, \
                    f"logits_k shape {kl_inputs['logits_k'].shape} batch/seq mismatch with log_probs {log_probs.shape}"
                if kl_inputs["topk_indices"] is not None:
                    assert kl_inputs["topk_indices"].shape == kl_inputs["logits_k"].shape, \
                        f"topk_indices shape {kl_inputs['topk_indices'].shape} != logits_k shape {kl_inputs['logits_k'].shape}"

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)

                # Derive logsumexp from log_probs
                logits_at_label = logits.gather(-1, micro_batch['responses'].unsqueeze(-1)).squeeze(-1)
                logsumexp = logits_at_label - log_probs

                if kl_topk_k == -1:
                    kl_inputs = {"logits_k": logits, "topk_indices": None, "logsumexp": logsumexp}
                elif kl_topk_k is not None and kl_topk_k > 0:
                    _, topk_indices = logits.topk(kl_topk_k, dim=-1)
                    logits_k = logits.gather(-1, topk_indices)
                    kl_inputs = {"logits_k": logits_k, "topk_indices": topk_indices, "logsumexp": logsumexp}
                else:
                    # kl_topk_k is None, gather at provided kl_topk_indices
                    assert kl_topk_indices is not None
                    # Validate kl_topk_indices shape
                    assert kl_topk_indices.shape[0] == batch_size, \
                        f"kl_topk_indices batch size {kl_topk_indices.shape[0]} != expected {batch_size}"
                    assert kl_topk_indices.shape[1] == response_length, \
                        f"kl_topk_indices response_length {kl_topk_indices.shape[1]} != expected {response_length}"
                    logits_k = logits.gather(-1, kl_topk_indices)
                    kl_inputs = {"logits_k": logits_k, "topk_indices": kl_topk_indices, "logsumexp": logsumexp}

                # Shape assertions for non-rmpad branch
                batch_size_out = log_probs.shape[0]
                seq_len_out = log_probs.shape[1]
                assert kl_inputs["logsumexp"].shape == (batch_size_out, seq_len_out), \
                    f"logsumexp shape {kl_inputs['logsumexp'].shape} != expected ({batch_size_out}, {seq_len_out})"
                assert kl_inputs["logits_k"].shape[0] == batch_size_out and kl_inputs["logits_k"].shape[1] == seq_len_out, \
                    f"logits_k shape {kl_inputs['logits_k'].shape} batch/seq mismatch with log_probs {log_probs.shape}"
                if kl_inputs["topk_indices"] is not None:
                    assert kl_inputs["topk_indices"].shape == kl_inputs["logits_k"].shape, \
                        f"topk_indices shape {kl_inputs['topk_indices'].shape} != logits_k shape {kl_inputs['logits_k'].shape}"

            return entropy, log_probs, kl_inputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_with_logits(self, data: DataProto, kl_topk_k: int) -> tuple:
        """Compute log probability and extract logits for KL divergence computation.

        Used for the initial forward pass (e.g., on the reference model) to get
        top-k indices, logits_k, and logsumexp.

        Args:
            data (DataProto): Input data containing input_ids, attention_mask, etc.
            kl_topk_k: -1 for full vocab logits, >0 for top-k logits.

        Returns:
            log_probs: tensor of shape [batch_size, response_length]
            entropys: tensor of shape [batch_size, response_length]
            kl_inputs: dict containing:
                - logits_k: tensor [batch_size, response_length, vocab_size or k]
                - topk_indices: tensor [batch_size, response_length, k] or None
                - logsumexp: tensor [batch_size, response_length]
        """
        assert kl_topk_k == -1 or kl_topk_k > 0, \
            f"kl_topk_k must be -1 (full logits) or >0 (top-k), got {kl_topk_k}"
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_list = []
        kl_inputs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, kl_inputs = self._forward_micro_batch_with_logits(
                    micro_batch, temperature=temperature, kl_topk_k=kl_topk_k
                )
            log_probs_lst.append(log_probs)
            entropy_list.append(entropy)
            kl_inputs_lst.append(kl_inputs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_list, dim=0)
        output_kl_inputs = {
            "logits_k": torch.concat([ki["logits_k"] for ki in kl_inputs_lst], dim=0),
            "logsumexp": torch.concat([ki["logsumexp"] for ki in kl_inputs_lst], dim=0),
        }
        if kl_topk_k > 0:
            output_kl_inputs["topk_indices"] = torch.concat(
                [ki["topk_indices"] for ki in kl_inputs_lst], dim=0
            )
        else:
            output_kl_inputs["topk_indices"] = None

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            entropys = entropys[revert_indices]
            output_kl_inputs["logits_k"] = output_kl_inputs["logits_k"][revert_indices]
            output_kl_inputs["logsumexp"] = output_kl_inputs["logsumexp"][revert_indices]
            if kl_topk_k > 0:
                output_kl_inputs["topk_indices"] = output_kl_inputs["topk_indices"][revert_indices]

        return log_probs, entropys, output_kl_inputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_at_indices(self, data: DataProto, topk_indices: torch.Tensor) -> tuple:
        """Compute log probability and gather logits at provided top-k indices.

        Used to gather the actor's (or ref's) logits at another model's top-k
        positions. For OPD with full_reverse KL: actor gathers at ref's top-k indices.

        Args:
            data (DataProto): Input data containing input_ids, attention_mask, etc.
            topk_indices: tensor of shape [batch_size, response_length, k].

        Returns:
            log_probs: tensor of shape [batch_size, response_length]
            kl_inputs: dict containing:
                - logits_k: tensor [batch_size, response_length, k]
                - topk_indices: tensor [batch_size, response_length, k] (same as input)
                - logsumexp: tensor [batch_size, response_length]
        """
        assert topk_indices.dim() == 3, \
            f"topk_indices must be 3D [batch, seq, k], got {topk_indices.dim()}D"
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
            topk_indices_chunks = topk_indices.chunk(num_micro_batches, dim=0)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            # Reorder topk_indices to match rearranged batches
            flat_indices = list(itertools.chain.from_iterable(indices))
            topk_indices_reordered = topk_indices[flat_indices]
            topk_indices_chunks = []
            start = 0
            for mb in micro_batches:
                mb_size = mb.batch_size[0] if hasattr(mb, "batch_size") else len(mb["input_ids"])
                topk_indices_chunks.append(topk_indices_reordered[start:start + mb_size])
                start += mb_size
        else:
            micro_batches = batch.split(micro_batch_size)
            topk_indices_chunks = list(torch.split(topk_indices, micro_batch_size, dim=0))

        log_probs_lst = []
        kl_inputs_lst = []
        for micro_batch, indices_split in zip(micro_batches, topk_indices_chunks):
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                _, log_probs, kl_inputs = self._forward_micro_batch_with_logits(
                    micro_batch, temperature=temperature, kl_topk_indices=indices_split
                )
            log_probs_lst.append(log_probs)
            kl_inputs_lst.append(kl_inputs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        output_kl_inputs = {
            "logits_k": torch.concat([ki["logits_k"] for ki in kl_inputs_lst], dim=0),
            "topk_indices": torch.concat([ki["topk_indices"] for ki in kl_inputs_lst], dim=0),
            "logsumexp": torch.concat([ki["logsumexp"] for ki in kl_inputs_lst], dim=0),
        }

        if use_dynamic_bsz:
            flat_indices_all = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(flat_indices_all), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            output_kl_inputs["logits_k"] = output_kl_inputs["logits_k"][revert_indices]
            output_kl_inputs["topk_indices"] = output_kl_inputs["topk_indices"][revert_indices]
            output_kl_inputs["logsumexp"] = output_kl_inputs["logsumexp"][revert_indices]

        return log_probs, output_kl_inputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_with_topk(self, data: DataProto, topk_k: int = 50) -> dict:
        """Compute log probabilities along with top-k token indices and their log probs.
        
        Used for OPD with true KL divergence calculation.
        
        Args:
            data: DataProto containing input_ids, attention_mask, position_ids, responses
            topk_k: number of top tokens to extract
            
        Returns:
            dict with keys:
                - log_probs: (batch_size, response_length) - log probs for selected tokens
                - topk_indices: (batch_size, response_length, k) - top-k token indices
                - topk_log_probs: (batch_size, response_length, k) - log probs for top-k tokens
        """
        from verl.utils.torch_functional import topk_logprobs_from_logits
        
        self.actor_module.eval()
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        
        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)
        
        log_probs_lst = []
        topk_indices_lst = []
        topk_log_probs_lst = []
        
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                # Run forward with topk extraction
                log_probs, topk_indices, topk_log_probs = self._forward_micro_batch_with_topk(
                    micro_batch, temperature=temperature, topk_k=topk_k
                )
            log_probs_lst.append(log_probs)
            topk_indices_lst.append(topk_indices)
            topk_log_probs_lst.append(topk_log_probs)
        
        log_probs = torch.concat(log_probs_lst, dim=0)
        topk_indices = torch.concat(topk_indices_lst, dim=0)
        topk_log_probs = torch.concat(topk_log_probs_lst, dim=0)
        
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            topk_indices = topk_indices[revert_indices]
            topk_log_probs = topk_log_probs[revert_indices]
        
        return {
            "log_probs": log_probs,
            "topk_indices": topk_indices,
            "topk_log_probs": topk_log_probs,
        }

    def _forward_micro_batch_with_topk(self, micro_batch, temperature, topk_k=50):
        """Forward pass that returns log probs for selected tokens AND top-k info.
        
        Follows the same rmpad pattern as _forward_micro_batch for memory efficiency.
        """
        from verl.utils.torch_functional import topk_logprobs_from_logits, logprobs_from_logits
        
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                # pad and slice if using ulysses sp
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                # Forward pass with rmpad
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad = logits_rmpad / temperature

                # Compute log probs for selected tokens
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=False,  # Can't use inplace since we need logits for top-k
                )

                # Compute top-k indices and log probs from rmpad logits
                topk_indices_rmpad, topk_log_probs_rmpad = topk_logprobs_from_logits(logits_rmpad, k=topk_k)
                # topk_indices_rmpad: (total_nnz, k)
                # topk_log_probs_rmpad: (total_nnz, k)

                # Gather if using ulysses sp
                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    # Gather top-k results
                    topk_indices_rmpad = gather_outpus_and_unpad(topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    topk_log_probs_rmpad = gather_outpus_and_unpad(topk_log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # Pad back to (bsz, seqlen)
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # Pad back top-k results: need to handle (total_nnz, k) -> (bsz, seqlen, k)
                full_topk_indices = pad_input(
                    hidden_states=topk_indices_rmpad,  # (total_nnz, k)
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )  # (bsz, seqlen, k)

                full_topk_log_probs = pad_input(
                    hidden_states=topk_log_probs_rmpad,  # (total_nnz, k)
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )  # (bsz, seqlen, k)

                # Extract response part
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]  # (bsz, response_length, k)
                topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]  # (bsz, response_length, k)

            else:  # not using rmpad
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits = output.logits
                logits = logits / temperature
                response_logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

                # Compute log probs and top-k
                log_probs = logprobs_from_logits(response_logits, micro_batch["responses"])
                topk_indices, topk_log_probs = topk_logprobs_from_logits(response_logits, k=topk_k)

            return log_probs, topk_indices, topk_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_for_topk_indices(self, data: DataProto, topk_indices: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for given top-k token indices.
        
        Used to compute student model's log probs for teacher's top-k tokens.
        
        Args:
            data: DataProto containing input_ids, attention_mask, position_ids, responses
            topk_indices: (batch_size, response_length, k) - indices to compute log probs for
            
        Returns:
            topk_log_probs: (batch_size, response_length, k) - log probs for given indices
        """
        from verl.utils.torch_functional import logprobs_from_logits_for_indices
        
        self.actor_module.eval()
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        
        batch_size = data.batch.batch_size[0]
        
        if has_multi_modal_inputs:
            num_micro_batches = batch_size // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
            topk_indices_chunks = topk_indices.chunk(num_micro_batches, dim=0)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            # Reorder topk_indices to match rearranged batches
            flat_indices = list(itertools.chain.from_iterable(indices))
            topk_indices_reordered = topk_indices[flat_indices]
            topk_indices_chunks = []
            start = 0
            for mb in micro_batches:
                mb_size = mb.batch_size[0] if hasattr(mb, 'batch_size') else len(mb['input_ids'])
                topk_indices_chunks.append(topk_indices_reordered[start:start + mb_size])
                start += mb_size
        else:
            micro_batches = batch.split(micro_batch_size)
            topk_indices_chunks = topk_indices.split(micro_batch_size, dim=0)
        
        topk_log_probs_lst = []
        
        for micro_batch, topk_idx_chunk in zip(micro_batches, topk_indices_chunks):
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                topk_log_probs = self._forward_micro_batch_for_indices(
                    micro_batch, topk_idx_chunk, temperature=temperature
                )
            topk_log_probs_lst.append(topk_log_probs)
        
        topk_log_probs = torch.concat(topk_log_probs_lst, dim=0)
        
        if use_dynamic_bsz:
            revert_indices = torch.tensor(get_reverse_idx(flat_indices), dtype=torch.long)
            topk_log_probs = topk_log_probs[revert_indices]
        
        return topk_log_probs

    def _forward_micro_batch_for_indices(self, micro_batch, topk_indices, temperature):
        """Forward pass that computes log probs for specific token indices.
        
        Follows rmpad pattern for memory efficiency.
        topk_indices: (bsz, response_length, k)
        """
        from verl.utils.torch_functional import logprobs_from_logits_for_indices
        
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                # unpad position_ids
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # topk_indices is (bsz, response_length, k) but indices are for (bsz, seqlen)
                # Expand to full seqlen first, then unpad.
                # Place at [-response_length-1:-1] because these are the logit positions
                # that PREDICT the response tokens (position t predicts token t+1).
                full_topk_indices = torch.zeros(batch_size, seqlen, topk_indices.shape[-1],
                                                dtype=topk_indices.dtype, device=topk_indices.device)
                full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                topk_indices_flat = rearrange(full_topk_indices, "b s k -> (b s) k")
                topk_indices_rmpad = index_first_axis(topk_indices_flat, indices)  # (total_nnz, k)

                # pad and slice if using ulysses sp
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    # Pad and slice topk_indices as well
                    topk_indices_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                        topk_indices_rmpad.unsqueeze(0),  # Add batch dim for padding
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    topk_indices_rmpad = topk_indices_rmpad.squeeze(0)  # Remove batch dim

                # Forward pass
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad = logits_rmpad / temperature

                # Compute log probs for given indices: (total_nnz, k)
                topk_log_probs_rmpad = logprobs_from_logits_for_indices(logits_rmpad, topk_indices_rmpad)

                # Gather if using ulysses sp
                if self.use_ulysses_sp:
                    topk_log_probs_rmpad = gather_outpus_and_unpad(
                        topk_log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # Pad back to (bsz, seqlen, k)
                full_topk_log_probs = pad_input(
                    hidden_states=topk_log_probs_rmpad,  # (total_nnz, k)
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )  # (bsz, seqlen, k)

                # Extract response part
                topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]  # (bsz, response_length, k)

            else:  # not using rmpad
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits = output.logits
                logits = logits / temperature
                response_logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

                # Compute log probs for given indices
                topk_log_probs = logprobs_from_logits_for_indices(response_logits, topk_indices)

            return topk_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            kl_loss_type = self.config.kl_loss_type
            kl_topk = self.config.get("kl_topk_tokens", None)
            if kl_loss_type in ("full_forward", "full_reverse"):
                # Full KL requires logits from reference model
                # For OPD full reverse KL, we always use ref_topk_indices here since teacher determines important tokens
                select_keys.extend(["ref_logits_k", "ref_logsumexp", "ref_topk_indices"])
                if self.config.get("kl_use_tail_sampling", False):
                    select_keys.append("ref_log_prob")
            else:
                # Token-level KL
                select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    use_full_kl = (self.config.use_kl_loss and self.config.kl_loss_type in ("full_forward", "full_reverse"))
                    kl_topk = self.config.get('kl_topk_tokens', None) if use_full_kl else None

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    if use_full_kl and kl_topk is not None and kl_topk > 0:
                        # Memory-efficient top-k mode: gather actor logits at ref's top-k indices
                        # For OPD: we ALWAYS use ref_topk_indices (teacher determines important tokens)
                        kl_topk_indices = data["ref_topk_indices"]

                        # if self.config.get("opd_mask_special_tokens", False):
                        #     id_endoftext = self.config.get("id_endoftext", 151643)
                        #     id_imend = self.config.get("id_im_end", 151645)
                        #     kl_topk_indices = kl_topk_indices.masked_fill(kl_topk_indices == id_endoftext, id_imend)

                        entropy, log_prob, actor_kl_inputs = self._forward_micro_batch_with_logits(
                            micro_batch=data, temperature=temperature,
                            kl_topk_indices=kl_topk_indices,
                        )
                        actor_logits_k = actor_kl_inputs["logits_k"]
                        actor_logsumexp = actor_kl_inputs["logsumexp"]
                    elif use_full_kl:
                        raise NotImplementedError("Full KL without top-k is not implemented yet due to memory constraints. Please set kl_topk_tokens > 0.")
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data, temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        if use_full_kl:
                            # Full KL divergence computation (ema-pg style)
                            use_kl_iw = self.config.get('use_kl_iw', False) # for importance sampling weight # 但我不太确定这个是否需要, 或许可以后续ablation一下?

                            if use_kl_iw: # Compute importance weight for off-policy correction
                                log_kl_iw = (log_prob - old_log_prob).detach()
                                log_kl_iw = torch.clamp(log_kl_iw, min=-20, max=20)
                                kl_iw = torch.exp(log_kl_iw)
                                # Apply optional clipping bounds
                                kl_iw_clip_lower = self.config.get('kl_iw_clip_lower', None)
                                kl_iw_clip_upper = self.config.get('kl_iw_clip_upper', None)
                                if kl_iw_clip_lower is not None or kl_iw_clip_upper is not None:
                                    kl_iw = torch.clamp(kl_iw, min=kl_iw_clip_lower, max=kl_iw_clip_upper)

                            if kl_topk is not None and kl_topk > 0:
                                # Memory-efficient top-k KL
                                use_tail_sampling = self.config.get("kl_use_tail_sampling", False) # 这是别人的创新点, 我们paper不追求刷分的话就不使用了
                                tail_kwargs = {}
                                if use_tail_sampling:
                                    # For OPD: use ref_topk_indices for the tail mask
                                    tail_kwargs = dict(
                                        actor_topk_indices=data["ref_topk_indices"],
                                        ref_topk_indices=data["ref_topk_indices"],
                                        sampled_indices=responses,
                                        log_prob=log_prob,
                                        ref_log_prob=data["ref_log_prob"],
                                    )
                                kl_L1, kl_L2 = compute_memory_efficient_kl(
                                    actor_logits_k=actor_logits_k,
                                    actor_logsumexp=actor_logsumexp,
                                    ref_logits_k=data["ref_logits_k"],
                                    ref_logsumexp=data["ref_logsumexp"],
                                    kl_type=kl_loss_type,
                                    use_tail_sampling=use_tail_sampling,
                                    norm_to_one_for_kl=self.config.get("norm_to_one_for_kl", True),
                                    clip_log_ratio=self.config.get("clip_log_ratio", False),
                                    **tail_kwargs,
                                )
                                if use_kl_iw:
                                    kl_L2 = kl_L2 * kl_iw
                                kld = kl_L1 + kl_L2
                            else:
                                raise NotImplementedError("Full KL without top-k is not implemented yet due to memory constraints. Please set kl_topk_tokens > 0.")
                        else:
                            # Token-level KL approximations
                            ref_log_prob = data["ref_log_prob"]
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )

                        # Apply OPD special token masking if enabled
                        kl_mask = self._compute_opd_kl_mask(responses, response_mask)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=kl_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    
                    # print(f"[DEBUG LOSS] loss: device={loss.device}, dtype={loss.dtype}, "
                    #     f"value={loss.item()}, is_nan={loss.isnan().any()}, is_inf={loss.isinf().any()}")
                    loss.backward()

                    data = {
                        # "actor/kl_loss": kl_loss.detach().item(),
                        "actor/policy_loss": policy_loss.detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
                self.actor_optimizer.zero_grad()
        
        return metrics
