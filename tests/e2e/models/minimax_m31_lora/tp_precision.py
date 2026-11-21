# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Two-rank HCCL projection controls using one common real input."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from low_rank_precision import errors, initialize_npu, read_weight
from safetensors.torch import load_file


def measure(args):
    rank = int(os.environ["LOCAL_RANK"])
    backend = initialize_npu(rank)
    torch.set_num_threads(8)
    dist.init_process_group("hccl")
    try:
        if dist.get_world_size() != 2:
            raise ValueError("Run with exactly two ranks")
        prefix = "language_model.model.layers.0.self_attn.o_proj"
        paths = sorted(args.snapshot_dir.glob(f"dp*-rank0-model.{prefix}.pt"))
        if len(paths) != 1:
            raise ValueError("Expected one active DP replica's TP0 input snapshot")
        values = [
            torch.load(
                paths[0].with_name(paths[0].name.replace("-rank0-", f"-rank{part}-")),
                map_location="cpu",
                weights_only=True,
            )["input0"][-4:]
            for part in range(2)
        ]
        x = torch.cat(values, dim=-1).npu()
        weight = read_weight(args.weights / "base", prefix + ".weight").npu()
        tensors = load_file(args.weights / f"adapter-{args.adapter}/adapter_model.safetensors")
        config = json.loads((args.weights / f"adapter-{args.adapter}/adapter_config.json").read_text())
        a = tensors["base_model.model." + prefix + ".lora_A.weight"].npu()
        b = tensors["base_model.model." + prefix + ".lora_B.weight"].npu() * (config["lora_alpha"] / config["r"])
        local_x = x.chunk(2, dim=-1)[rank].contiguous()
        local_w = weight.chunk(2, dim=-1)[rank].contiguous()
        local_a = a.chunk(2, dim=-1)[rank].contiguous()
        full_base = torch.nn.functional.linear(x, weight)
        full_shrink = x.float() @ a.float().T
        full_low_rank = (full_base.float() + full_shrink @ b.float().T).bfloat16()
        full_precise = (x.float() @ weight.float().T + full_shrink @ b.float().T).bfloat16()
        partial_base = torch.nn.functional.linear(local_x, local_w)
        shrink = local_x.float() @ local_a.float().T
        partial_low_rank = (partial_base.float() + shrink @ b.float().T).bfloat16()
        partial_precise = local_x.float() @ local_w.float().T + shrink @ b.float().T
        dist.all_reduce(partial_base)
        dist.all_reduce(partial_low_rank)
        dist.all_reduce(partial_precise)
        dist.all_reduce(shrink)
        post_reduce = (partial_base.float() + shrink @ b.float().T).bfloat16()
        result = {
            **backend,
            "world_size": dist.get_world_size(),
            "backend": dist.get_backend(),
            "input_shape": list(x.shape),
            "base_tp2_vs_tp1": errors(partial_base, full_base),
            "low_rank_tp2_vs_tp1": errors(partial_low_rank, full_low_rank),
            "post_reduce_tp2_vs_tp1": errors(post_reduce, full_low_rank),
            "fp32_partials_tp2_vs_tp1": errors(partial_precise.bfloat16(), full_precise),
        }
        if rank == 0:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--adapter", choices=("a", "b"), default="b")
    parser.add_argument("--output", required=True, type=Path)
    measure(parser.parse_args())
