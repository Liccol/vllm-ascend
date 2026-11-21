# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Attention LoRA for MiniMax M3's fused QKV/indexer projection."""

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import MergedQKVParallelLinearWithLoRA
from vllm.lora.layers.utils import _not_fully_sharded_can_replace


class AscendMiniMaxM3QKVParallelLinearWithLoRA(MergedQKVParallelLinearWithLoRA):
    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        # The base exposes five outputs; the adapter mapping covers Q/K/V.
        self.n_slices = len(self.output_slices)
        self.output_sizes = base_layer.output_sizes[: self.n_slices]
        self.output_size = sum(self.output_slices)

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        # Keep the custom base projection on every TP rank.
        return self._apply_base_forward(x)

    def _apply_lora_to_output(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # Punica requires a contiguous QKV matrix, not a five-slice row stride.
        qkv = output[..., : self.output_size].contiguous()
        output[..., : self.output_size] = super()._apply_lora_to_output(x, qkv)
        return output

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        if type(source_layer).__name__ != "AscendMiniMaxM3QKVParallelLinearWithIndexer":
            return False
        # Avoid importing model/attention backends for unrelated model classes.
        from vllm_ascend.models.minimax_m3.msa_m3 import AscendMiniMaxM3QKVParallelLinearWithIndexer

        return type(source_layer) is AscendMiniMaxM3QKVParallelLinearWithIndexer and packed_modules_list == [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
