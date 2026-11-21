# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Independent real-input LoRA and TP arithmetic controls, not serving kernels."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def initialize_npu(device):
    import torch_npu  # noqa: F401

    from vllm_ascend.batch_invariant import HAS_ASCENDC_BATCH_INVARIANT, enable_batch_invariant_mode

    torch.npu.set_device(device)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.set_deterministic_level(3)
    enable_batch_invariant_mode()
    return {"batch_invariant": True, "ascendc_batch_invariant": HAS_ASCENDC_BATCH_INVARIANT}


def errors(actual, expected):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    difference = actual - expected
    return {
        "equal": torch.equal(actual, expected),
        "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-30)).item(),
        "max_absolute": difference.abs().max().item(),
        "unequal_elements": (actual != expected).sum().item(),
        "elements": expected.numel(),
    }


def read_weight(directory, key):
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(directory / index[key], framework="pt") as handle:
        return handle.get_tensor(key)


def measure(weights, snapshots, adapter, device):
    tensors = load_file(weights / f"adapter-{adapter}/adapter_model.safetensors")
    config = json.loads((weights / f"adapter-{adapter}/adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    rows = {}
    for projection in ("q_proj", "o_proj"):
        prefix = "language_model.model.layers.0.self_attn."
        base = read_weight(weights / "base", prefix + projection + ".weight")
        merged = read_weight(weights / f"merged-{adapter}", prefix + projection + ".weight")
        a = tensors["base_model.model." + prefix + projection + ".lora_A.weight"]
        b = tensors["base_model.model." + prefix + projection + ".lora_B.weight"] * scale
        suffix = "qkv_proj" if projection == "q_proj" else projection
        paths = sorted(snapshots.glob(f"dp*-rank0-model.{prefix}{suffix}.pt"))
        if len(paths) != 1:
            raise ValueError("Expected one active DP replica's TP0 input snapshot")
        x = torch.load(paths[0], map_location="cpu", weights_only=True)["input0"]
        if projection == "o_proj":
            other = paths[0].with_name(paths[0].name.replace("-rank0-", "-rank1-"))
            x = torch.cat([x, torch.load(other, map_location="cpu", weights_only=True)["input0"]], dim=-1)
        x = x[-4:].contiguous()
        if not torch.equal((base.float() + b.float() @ a.float()).bfloat16(), merged):
            raise ValueError("Saved BF16 merged weights disagree with the adapter formula")
        ideal_delta = b.double() @ a.double()
        effective_delta = merged.double() - base.double()
        block = effective_delta[:64, :64]
        singular = torch.linalg.svdvals(block)
        ideal = x.double() @ base.double().T + (x.double() @ a.double().T) @ b.double().T
        merged_oracle = (x.double() @ merged.double().T).bfloat16()
        base_oracle = (x.double() @ base.double().T).bfloat16()
        low_rank_oracle = (base_oracle.double() + (x.double() @ a.double().T) @ b.double().T).bfloat16()
        row = {
            "input_shape": list(x.shape),
            "ideal_rank": config["r"],
            "effective_update_block_rank": torch.linalg.matrix_rank(block).item(),
            "best_rank8_block_relative_residual": (singular[8:].norm() / singular.norm()).item(),
            "weight_rounding": errors(effective_delta, ideal_delta),
            "fp64_low_rank_with_bf16_base_vs_merged": errors(low_rank_oracle, merged_oracle),
            "fp64_single_output_rounding_vs_merged": errors(ideal.bfloat16(), merged_oracle),
        }
        x, base, merged, a, b = [value.to(device) for value in (x, base, merged, a, b)]
        full_base = torch.nn.functional.linear(x, base)
        delta = (x.float() @ a.float().T) @ b.float().T
        low_rank = (full_base.float() + delta).bfloat16()
        dense = torch.nn.functional.linear(x, merged)
        precise = (x.float() @ base.float().T + delta).bfloat16()
        row["npu_low_rank_vs_merged"] = errors(low_rank, dense)
        row["npu_fp32_single_output_rounding_vs_merged"] = errors(precise, dense)
        row["npu_merged_vs_fp64_oracle"] = errors(dense.cpu(), merged_oracle)
        if projection == "q_proj":
            partitioned = torch.cat([torch.nn.functional.linear(x, shard) for shard in base.chunk(2, dim=0)], dim=-1)
            row["tp2_simulated_column_vs_tp1_base"] = errors(partitioned, full_base)
        if projection == "o_proj":
            inputs, bases, factors = x.chunk(2, dim=-1), base.chunk(2, dim=-1), a.chunk(2, dim=-1)
            partial_bases = [torch.nn.functional.linear(value, weight) for value, weight in zip(inputs, bases)]
            base_tp2 = partial_bases[0] + partial_bases[1]
            shrinks = [value.float() @ factor.float().T for value, factor in zip(inputs, factors)]
            partials = [
                (plain.float() + shrink @ b.float().T).bfloat16() for plain, shrink in zip(partial_bases, shrinks)
            ]
            partial_fp32 = [
                value.float() @ weight.float().T + shrink @ b.float().T
                for value, weight, shrink in zip(inputs, bases, shrinks)
            ]
            row["tp2_simulated_base_vs_tp1_base"] = errors(base_tp2, full_base)
            row["tp2_simulated_low_rank_vs_tp1_low_rank"] = errors(partials[0] + partials[1], low_rank)
            post_reduce = (base_tp2.float() + (shrinks[0] + shrinks[1]) @ b.float().T).bfloat16()
            row["tp2_post_reduce_vs_tp1_low_rank"] = errors(post_reduce, low_rank)
            row["tp2_fp32_partials_vs_tp1_fp32"] = errors((partial_fp32[0] + partial_fp32[1]).bfloat16(), precise)
        rows[projection] = row
    return {
        "adapter": adapter,
        "device": device,
        "simulated_tp_note": "Two local GEMMs plus addition, not HCCL",
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--adapter", choices=("a", "b"), default="b")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    backend = initialize_npu(0)
    torch.set_num_threads(8)
    result = measure(args.weights, args.snapshot_dir, args.adapter, "npu:0")
    result.update(backend)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
