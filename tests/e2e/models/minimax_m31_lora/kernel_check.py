# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Diagnose the actual NPU low-rank branch independently of the model body."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401
from safetensors import safe_open
from safetensors.torch import load_file
from vllm.config import LoRAConfig
from vllm.lora.layers import LoRAMapping

from vllm_ascend.utils import adapt_patch, enable_custom_op, register_ascend_customop


def diagnose():
    adapt_patch()
    adapt_patch(True)
    register_ascend_customop()
    enable_custom_op()
    # Initialize in the same order as a worker to avoid MoE import cycles.
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    torch.npu.set_device(0)
    torch.manual_seed(42)
    sizes = (4096, 256, 256)
    x = torch.randn(6, 6144, dtype=torch.bfloat16).npu()
    a = tuple((torch.randn(2, 1, 8, 6144) * 0.1).bfloat16().npu() for _ in sizes)
    b = tuple((torch.randn(2, 1, size, 8) * 0.2).bfloat16().npu() for size in sizes)
    ids = [1, 2, 0, 1, 2, 0]
    punica = PunicaWrapperNPU(
        64, 64, "npu:0", lora_config=LoRAConfig(max_loras=2, max_lora_rank=8, lora_dtype=torch.bfloat16)
    )

    def execute(values, mapping):
        punica.update_metadata(LoRAMapping(tuple(mapping), tuple(mapping), is_prefill=True), [1, 2], 3, 200064)
        output = torch.zeros(values.shape[0], sum(sizes), dtype=torch.bfloat16, device=values.device)
        punica.add_lora_linear(output, values, a, b, 1.0, sizes)
        return output.cpu()

    mixed = execute(x, ids)
    single = torch.cat([execute(x[i : i + 1], [slot]) for i, slot in enumerate(ids)])
    reference = torch.zeros_like(mixed)
    for i, adapter in enumerate(ids):
        offset = 0
        if adapter:
            for aw, bw, size in zip(a, b, sizes):
                reference[i, offset : offset + size] = (
                    (x[i].cpu().float() @ aw[adapter - 1, 0].cpu().float().T) @ bw[adapter - 1, 0].cpu().float().T
                ).bfloat16()
                offset += size
    return {
        "mixed_vs_single_equal": torch.equal(mixed, single),
        "mixed_vs_single_max_error": (mixed.float() - single.float()).abs().max().item(),
        "reference_relative_l2": ((mixed.float() - reference.float()).norm() / reference.float().norm()).item(),
        "base_rows_zero": bool((mixed[[2, 5]] == 0).all()),
    }


def measure_real_projection(weights: Path):
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    name = "language_model.model.layers.0.self_attn.q_proj.weight"

    def read(model):
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(model / index[name], framework="pt") as handle:
            return handle.get_tensor(name)[:4096].contiguous()

    base_weight, merged_weight = read(weights / "base"), read(weights / "merged-a")
    tensors = load_file(weights / "adapter-a/adapter_model.safetensors")
    prefix = "base_model.model." + name.removesuffix(".weight")
    a = tensors[prefix + ".lora_A.weight"]
    b = tensors[prefix + ".lora_B.weight"][:4096].contiguous()
    config = json.loads((weights / "adapter-a/adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    rebuilt = (base_weight.float() + scale * (b.float() @ a.float())).bfloat16()
    if not torch.equal(rebuilt, merged_weight):
        raise ValueError("Saved merged weights disagree with the recorded merge formula")
    torch.manual_seed(42)
    x = torch.randn(6, a.shape[1], dtype=torch.bfloat16).npu()
    lora_a = torch.zeros(2, 1, *a.shape, dtype=torch.bfloat16, device=x.device)
    lora_b = torch.zeros(2, 1, *b.shape, dtype=torch.bfloat16, device=x.device)
    lora_a[0, 0].copy_(a)
    lora_b[0, 0].copy_(b * scale)
    output = torch.nn.functional.linear(x, base_weight.npu())
    plain = output.cpu().float()
    punica = PunicaWrapperNPU(
        64, 64, "npu:0", lora_config=LoRAConfig(max_loras=2, max_lora_rank=8, lora_dtype=torch.bfloat16)
    )
    punica.update_metadata(LoRAMapping((1,) * 6, (1,) * 6, is_prefill=True), [1, 2], 3, 200064)
    punica.add_lora_linear(output, x, (lora_a,), (lora_b,), 1.0, (4096,))
    merged_output = torch.nn.functional.linear(x, merged_weight.npu()).cpu()
    delta = (x.cpu().float() @ a.float().T) @ (b.float() * scale).T
    oracle = (plain + delta).bfloat16()
    actual = output.cpu()
    return {
        "merge_matches_saved_weights": True,
        "lora_vs_merged_bitwise_equal": torch.equal(actual, merged_output),
        "lora_vs_merged_relative_l2": (
            (actual.float() - merged_output.float()).norm() / merged_output.float().norm()
        ).item(),
        "lora_vs_unmerged_oracle_relative_l2": (
            (actual.float() - oracle.float()).norm() / oracle.float().norm()
        ).item(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path, help="Optional prepared real BF16 fixture root")
    args = parser.parse_args()
    result = diagnose()
    if args.weights:
        result["real_projection"] = measure_real_projection(args.weights)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(not result["mixed_vs_single_equal"] or not result["base_rows_zero"])
