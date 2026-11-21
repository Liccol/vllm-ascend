# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Merged LoRA routing and slot lifecycle, independently of NPU serving."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.ut.lora.test_minimax_m3 import FiveSliceProjection
from vllm_ascend.lora.minimax_m3_merged import (
    AscendMiniMaxM3MergedQKVWithLoRA,
    AscendMiniMaxM3MergedRowWithLoRA,
)
from vllm_ascend.ops.linear import AscendQKVParallelLinear, AscendRowParallelLinear


@pytest.fixture(autouse=True)
def cpu_weights(monkeypatch):
    monkeypatch.setattr("vllm_ascend.lora.minimax_m3_merged.maybe_trans_nz", lambda value: value)


def mapping(slots):
    return SimpleNamespace(no_lora=all(slot < 0 for slot in slots), token_lora_indices=torch.tensor(slots))


def config():
    return SimpleNamespace(max_lora_rank=8, lora_dtype=torch.bfloat16, fully_sharded_loras=False)


@pytest.mark.parametrize("tp_size,kv_heads", [(1, 2), (2, 2), (4, 2)])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("with_indexer", [False, True])
def test_merged_qkv_routes_base_and_adapters_and_reuses_slots(tp_size, kv_heads, batched, with_indexer):
    torch.manual_seed(7)
    slots = [-1, 0, 1, 1, 0, -1]
    x = torch.randn((2, 3, 4) if batched else (6, 4), dtype=torch.bfloat16)
    adapters = [
        (
            [torch.randn(2, 4).bfloat16() for _ in range(3)],
            [torch.randn(rows, 2).bfloat16() for rows in (8, kv_heads * 2, kv_heads * 2)],
        )
        for _ in range(2)
    ]
    for rank in range(tp_size):
        base = FiveSliceProjection(tp_size, rank, kv_heads, torch.bfloat16)
        if not with_indexer:
            base.output_sizes = base.output_sizes[:3]
            base.output_size_per_partition -= 6
            base.weight = nn.Parameter(base.weight[:-6].detach().clone())
            base.bias = nn.Parameter(base.bias[:-6].detach().clone())
        original = base.weight.detach().clone()
        layer = AscendMiniMaxM3MergedQKVWithLoRA(base)
        layer.create_lora_weights(2, config())
        pointers = [weight.data_ptr() for weight in layer.merged_weights]
        layer.set_mapping(mapping(slots))
        merged = []
        for slot, (a, b) in enumerate(adapters):
            expected_weight = original.clone()
            offset = 0
            for i, size in enumerate(layer.output_slices):
                shard = rank if i == 0 else rank // max(1, tp_size // kv_heads)
                delta = b[i][shard * size : (shard + 1) * size].float() @ a[i].float()
                expected_weight[offset : offset + size] = (original[offset : offset + size].float() + delta).bfloat16()
                offset += size
            layer.set_lora(slot, a, b)
            merged.append(expected_weight)
        expected = torch.stack(
            [
                nn.functional.linear(value, original if slot < 0 else merged[slot], base.bias)
                for value, slot in zip(x.reshape(6, 4), slots)
            ]
        ).reshape(*x.shape[:-1], -1)
        actual = layer(x)[0]
        assert torch.equal(actual, expected)
        assert torch.equal(actual[..., sum(layer.output_slices) :], base(x)[0][..., sum(layer.output_slices) :])
        a, b = adapters[0]
        layer.set_lora(0, [a[0], None, None], [b[0], None, None])
        partial = layer(x)[0].reshape(6, -1)
        plain = base(x)[0].reshape(6, -1)
        assert torch.equal(partial[[1, 4], layer.output_slices[0] :], plain[[1, 4], layer.output_slices[0] :])
        assert torch.equal(partial[[2, 3]], actual.reshape(6, -1)[[2, 3]])
        for slot in (0, 1):
            layer.reset_lora(slot)
        assert torch.equal(layer(x)[0], base(x)[0])
        layer.set_lora(0, [torch.zeros_like(value) for value in a], b)
        assert not any(layer.merged_enabled)
        assert torch.equal(layer(x)[0], base(x)[0])
        assert pointers == [weight.data_ptr() for weight in layer.merged_weights]
        assert torch.equal(base.weight, original)


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("skip_bias", [False, True])
def test_merged_row_uses_input_shard_and_rank_zero_bias(tp_rank, skip_bias):
    torch.manual_seed(3)
    base = AscendRowParallelLinear.__new__(AscendRowParallelLinear)
    nn.Module.__init__(base)
    base.input_size, base.input_size_per_partition, base.output_size = 8, 4, 6
    base.tp_size, base.tp_rank = 2, tp_rank
    base.input_is_parallel, base.reduce_results, base.return_bias = True, False, True
    base.skip_bias_add = skip_bias
    base.weight = nn.Parameter(torch.randn(6, 4).bfloat16())
    base.bias = nn.Parameter(torch.randn(6).bfloat16())
    base.quant_method = SimpleNamespace(apply=lambda layer, x, bias: nn.functional.linear(x, layer.weight, bias))
    layer = AscendMiniMaxM3MergedRowWithLoRA(base)
    layer.create_lora_weights(2, config())
    layer.set_mapping(mapping([0, -1, 1]))
    a, b = torch.randn(2, 8).bfloat16(), torch.randn(6, 2).bfloat16()
    layer.set_lora(0, a, b)
    layer.set_lora(1, a * 2, b)
    x = torch.randn(3, 4).bfloat16()
    actual, bias = layer(x)
    local_a = a[:, tp_rank * 4 : (tp_rank + 1) * 4].float()
    expected = []
    effective_bias = base.bias if not skip_bias and tp_rank == 0 else None
    for value, scale in zip(x, (1, 0, 2)):
        weight = (base.weight.float() + scale * (b.float() @ local_a)).bfloat16()
        expected.append(nn.functional.linear(value, weight, effective_bias))
    assert torch.equal(actual, torch.stack(expected))
    assert bias is (base.bias if skip_bias else None)


def test_replacement_is_opt_in_and_limited_to_attention(monkeypatch):
    mode = SimpleNamespace(lora_execution_mode="low_rank")
    monkeypatch.setattr("vllm_ascend.lora.minimax_m3_merged.get_ascend_config", lambda: mode)
    lora = SimpleNamespace(fully_sharded_loras=False)
    model = SimpleNamespace(architectures=["MiniMaxM3SparseForConditionalGeneration"])
    for source_type, wrapper, suffix, targets in (
        (AscendQKVParallelLinear, AscendMiniMaxM3MergedQKVWithLoRA, "qkv_proj", ["q_proj", "k_proj", "v_proj"]),
        (AscendRowParallelLinear, AscendMiniMaxM3MergedRowWithLoRA, "o_proj", []),
    ):
        source = source_type.__new__(source_type)
        nn.Module.__init__(source)
        source.prefix = f"language_model.model.layers.0.self_attn.{suffix}"

        def matches(wrapper=wrapper, source=source, targets=targets):
            return wrapper.can_replace_layer(
                source_layer=source, lora_config=lora, packed_modules_list=targets, model_config=model
            )

        mode.lora_execution_mode = "low_rank"
        assert not matches()
        mode.lora_execution_mode = "merged"
        assert matches()
        source.prefix = "language_model.model.layers.0.mlp.down_proj"
        assert not matches()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_merged_cache_rejects_non_bf16_weights(dtype):
    with pytest.raises(ValueError, match="BF16 target weights"):
        AscendMiniMaxM3MergedQKVWithLoRA(FiveSliceProjection(1, 0, 2, dtype))
