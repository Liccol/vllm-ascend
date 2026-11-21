# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Projection reference tests; real NPU serving is tested separately."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.lora.minimax_m3 import AscendMiniMaxM3QKVParallelLinearWithLoRA


class FiveSliceProjection(nn.Module):
    def __init__(self, tp_size, tp_rank, kv_heads, dtype):
        super().__init__()
        self.input_size = 4
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self.head_size = 2
        self.num_heads = 4 // tp_size
        self.num_kv_heads = max(1, kv_heads // tp_size)
        self.num_kv_head_replicas = max(1, tp_size // kv_heads)
        q, kv = self.num_heads * 2, self.num_kv_heads * 2
        self.output_sizes = [q * tp_size, kv * tp_size, kv * tp_size, 4 * tp_size, 2 * tp_size]
        self.output_size_per_partition = q + 2 * kv + 6
        self.weight = nn.Parameter(torch.randn(self.output_size_per_partition, 4, dtype=dtype))
        self.bias = nn.Parameter(torch.randn(self.output_size_per_partition, dtype=dtype))
        self.skip_bias_add = False
        self.gather_output = False
        self.return_bias = True
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return nn.functional.linear(x, self.weight, None if self.skip_bias_add else self.bias), (
            self.bias if self.skip_bias_add else None
        )


class ReferencePunica:
    def __init__(self, slots):
        self.slots = slots

    def add_lora_linear(self, output, x, weights_a, weights_b, scale, output_slices):
        assert output.is_contiguous()
        for token, slot in enumerate(self.slots):
            if slot < 0:
                continue
            offset = 0
            for a, b, size in zip(weights_a, weights_b, output_slices):
                delta = (x[token].float() @ a[slot, 0].float().T) @ b[slot, 0].float().T
                output[token, offset : offset + size] += (delta * scale).to(output.dtype)
                offset += size
        return output


@pytest.mark.parametrize("tp_size,kv_heads", [(1, 2), (2, 2), (2, 4), (4, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batched", [False, True])
def test_base_and_two_adapters_preserve_indexer(tp_size, kv_heads, dtype, batched):
    torch.manual_seed(42)
    adapters = [
        (
            [torch.randn(2, 4, dtype=dtype) for _ in range(3)],
            [torch.randn(rows, 2, dtype=dtype) * 0.5 for rows in (8, kv_heads * 2, kv_heads * 2)],
        )
        for _ in range(2)
    ]
    slots = [-1, 0, 1, 1, 0, -1]
    x = torch.randn((2, 3, 4) if batched else (6, 4), dtype=dtype)
    for rank in range(tp_size):
        base = FiveSliceProjection(tp_size, rank, kv_heads, dtype)
        layer = AscendMiniMaxM3QKVParallelLinearWithLoRA(base)
        config = SimpleNamespace(max_lora_rank=8, lora_dtype=dtype, fully_sharded_loras=False)
        layer.create_lora_weights(2, config)
        layer.set_mapping(ReferencePunica(slots))
        assert layer.n_slices == len(layer.lora_a_stacked) == len(layer.lora_b_stacked) == 3
        expected = base(x)[0]
        tail = expected[..., sum(layer.output_slices) :].clone()
        for slot, (a, b) in enumerate(adapters):
            layer.set_lora(slot, a, b)
        flat_x, flat_expected = x.reshape(-1, 4), expected.reshape(6, -1)
        for token, slot in enumerate(slots):
            if slot < 0:
                continue
            a, b = adapters[slot]
            offset = 0
            for i, size in enumerate(layer.output_slices):
                delta = (flat_x[token].float() @ a[i].float().T) @ b[i].float().T
                shard = rank if i == 0 else rank // max(1, tp_size // kv_heads)
                flat_expected[token, offset : offset + size] += delta[shard * size : (shard + 1) * size].to(dtype)
                offset += size
        calls = base.calls
        actual, bias = layer(x)
        assert base.calls == calls + 1
        assert bias is None
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert torch.equal(actual[..., sum(layer.output_slices) :], tail)
        # Reuse one slot with Q-only weights; K/V and the other slot must survive correctly.
        a, b = adapters[0]
        layer.set_lora(0, [a[0], None, None], [b[0], None, None])
        partial = layer(x)[0].reshape(6, -1)
        plain = base(x)[0].reshape(6, -1)
        for token in (1, 4):
            assert torch.equal(partial[token, layer.output_slices[0] :], plain[token, layer.output_slices[0] :])
        for token in (2, 3):
            torch.testing.assert_close(partial[token], flat_expected[token], rtol=1e-5, atol=1e-6)
        layer.reset_lora(0)
        layer.reset_lora(1)
        torch.testing.assert_close(layer(x)[0], base(x)[0], rtol=0, atol=0)


def test_skip_bias_add_contract():
    base = FiveSliceProjection(1, 0, 2, torch.bfloat16)
    base.skip_bias_add = True
    layer = AscendMiniMaxM3QKVParallelLinearWithLoRA(base)
    config = SimpleNamespace(max_lora_rank=8, lora_dtype=torch.bfloat16, fully_sharded_loras=False)
    layer.create_lora_weights(2, config)
    layer.set_mapping(ReferencePunica([-1, -1, -1]))
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    output, bias = layer(x)
    assert bias is base.bias
    torch.testing.assert_close(output, nn.functional.linear(x, base.weight), rtol=0, atol=0)


def test_replacement_accepts_multiple_slots_only_for_m3_qkv():
    from vllm_ascend.models.minimax_m3.msa_m3 import AscendMiniMaxM3QKVParallelLinearWithIndexer

    source = AscendMiniMaxM3QKVParallelLinearWithIndexer.__new__(AscendMiniMaxM3QKVParallelLinearWithIndexer)
    nn.Module.__init__(source)
    config = SimpleNamespace(max_loras=2, fully_sharded_loras=False)

    def matches(layer, packed):
        return AscendMiniMaxM3QKVParallelLinearWithLoRA.can_replace_layer(
            source_layer=layer, lora_config=config, packed_modules_list=packed
        )

    targets = ["q_proj", "k_proj", "v_proj"]
    assert matches(source, targets)
    assert not matches(nn.Linear(4, 4), targets)
    assert not matches(source, ["index_q_proj", "index_k_proj"])
    config.fully_sharded_loras = True
    assert not matches(source, targets)
