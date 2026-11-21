# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in BF16 merged target layers for MiniMax M3 attention adapters."""

import torch
from torch import nn
from vllm.logger import logger
from vllm.lora.layers import RowParallelLinearWithLoRA
from vllm.lora.layers.utils import _not_fully_sharded_can_replace

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.lora.minimax_m3 import AscendMiniMaxM3QKVParallelLinearWithLoRA
from vllm_ascend.ops.linear import AscendQKVParallelLinear, AscendRowParallelLinear
from vllm_ascend.utils import maybe_trans_nz


def _is_minimax_m3(model_config):
    architectures = getattr(model_config, "architectures", ()) or ()
    return any(
        name in ("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration") for name in architectures
    )


class _MergedAttentionLoRAMixin:
    def __init__(self, base_layer):
        super().__init__(base_layer)
        if base_layer.weight.dtype != torch.bfloat16 or base_layer.weight.ndim != 2:
            raise ValueError("Merged attention LoRA requires unquantized BF16 target weights")

    def create_lora_weights(self, max_loras, lora_config, model_config=None):
        super().create_lora_weights(max_loras, lora_config, model_config)
        if self.lora_a_stacked[0].dtype != torch.bfloat16:
            raise ValueError("Merged attention LoRA requires BF16 adapter weights")
        # Reserve the maximum cache before profiling, including zero-adapter warmup.
        self.merged_weights = tuple(maybe_trans_nz(torch.empty_like(self.base_layer.weight)) for _ in range(max_loras))
        self.merged_enabled = [False] * max_loras
        logger.info(
            "%s reserved %d bytes for %d merged attention LoRA slots",
            type(self).__name__,
            sum(weight.numel() * weight.element_size() for weight in self.merged_weights),
            max_loras,
        )

    def reset_lora(self, index):
        self.merged_enabled[index] = False
        super().reset_lora(index)

    def set_lora(self, index, lora_a, lora_b):
        super().set_lora(index, lora_a, lora_b)
        a_inputs = lora_a if isinstance(lora_a, list) else [lora_a]
        b_inputs = lora_b if isinstance(lora_b, list) else [lora_b]
        # Dummy DP execution repeatedly installs zero adapters. Their result is the base.
        if not any(
            a is not None and b is not None and bool(a.detach().cpu().any()) and bool(b.detach().cpu().any())
            for a, b in zip(a_inputs, b_inputs)
        ):
            return
        merged = self.base_layer.weight.detach().cpu().clone()
        offset = 0
        for a, b, size in zip(self.lora_a_stacked, self.lora_b_stacked, self.output_slices):
            a = a[index, 0].detach().cpu().float()
            b = b[index, 0, :size].detach().cpu().float()
            merged[offset : offset + size] = (merged[offset : offset + size].float() + b @ a).bfloat16()
            offset += size
        self.merged_weights[index].copy_(merged, non_blocking=False)
        self.merged_enabled[index] = True

    def _apply_lora_to_output(self, x, output):
        if self.punica_wrapper.no_lora:
            return output
        shape = output.shape
        x = x.reshape(-1, x.shape[-1])
        output = output.reshape(-1, output.shape[-1])
        slots = self.punica_wrapper.token_lora_indices
        if slots.shape[0] != x.shape[0]:
            raise ValueError("Merged attention LoRA rows differ from the token mapping")
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        if isinstance(self, RowParallelLinearWithLoRA) and self.tp_rank > 0:
            bias = None
        for slot, enabled in enumerate(self.merged_enabled):
            if enabled:
                candidate = nn.functional.linear(x, self.merged_weights[slot], bias)
                output = torch.where((slots == slot).unsqueeze(-1), candidate, output)
        return output.reshape(shape)


class AscendMiniMaxM3MergedQKVWithLoRA(_MergedAttentionLoRAMixin, AscendMiniMaxM3QKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None):
        if not _is_minimax_m3(model_config) or get_ascend_config().lora_execution_mode != "merged":
            return False
        if type(source_layer) is AscendQKVParallelLinear:
            return source_layer.prefix.endswith(".self_attn.qkv_proj") and packed_modules_list == [
                "q_proj",
                "k_proj",
                "v_proj",
            ]
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        )


class AscendMiniMaxM3MergedRowWithLoRA(_MergedAttentionLoRAMixin, RowParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None):
        return (
            type(source_layer) is AscendRowParallelLinear
            and source_layer.prefix.endswith(".self_attn.o_proj")
            and _is_minimax_m3(model_config)
            and get_ascend_config().lora_execution_mode == "merged"
        )
