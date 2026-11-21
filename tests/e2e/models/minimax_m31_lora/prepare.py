# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Build a real-weight reduced BF16 base, two synthetic adapters and merged references."""

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
ADAPTER_STD = 0.1
LAYER_PREFIX = "language_model.model.layers."


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prepare(source: Path, output: Path, layers: int, rank: int, alpha: int):
    """Stream retained tensors, then derive both adapter and merged paths from the same base."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("An unquantized BF16 checkpoint is required")
    if not 1 <= layers <= text["num_hidden_layers"]:
        raise ValueError("Invalid retained layer count")
    indexes = list(source.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError("Expected exactly one safetensors index")
    index = json.loads(indexes[0].read_text())["weight_map"]
    selected = {}
    for name, shard in index.items():
        if name.startswith(LAYER_PREFIX):
            layer_index = int(name.removeprefix(LAYER_PREFIX).split(".", 1)[0])
            if layer_index >= layers:
                continue
        if ".mtp." in name or ".dspark." in name:
            continue
        selected[name] = shard
    target_keys = [f"language_model.model.layers.{i}.self_attn.{p}.weight" for i in range(layers) for p in TARGETS]
    missing = set(target_keys) - selected.keys()
    if missing:
        raise ValueError(f"Missing attention weights: {sorted(missing)}")
    base = output / "base"
    base.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".py", ".txt", ".model", ".jinja"):
            if "index.json" not in path.name:
                shutil.copy2(path, base / path.name)
    text["num_hidden_layers"] = layers
    for key in ("layer_types", "mlp_layer_types", "moe_layer_freq"):
        if isinstance(text.get(key), list):
            text[key] = text[key][:layers]
    sparse = text.get("sparse_attention_config", {})
    for key in ("sparse_attention_freq", "sparse_disable_index_value"):
        if isinstance(sparse.get(key), list):
            sparse[key] = sparse[key][:layers]
    write_json(base / "config.json", config)
    grouped = defaultdict(list)
    for name, shard in selected.items():
        grouped[shard].append(name)
    weight_map, shapes, total_size = {}, {}, 0
    for number, (shard, names) in enumerate(sorted(grouped.items())):
        filename = f"model-{number:05d}.safetensors"
        tensors = {}
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            for name in names:
                view = handle.get_slice(name)
                if view.get_dtype() not in ("BF16", "F32"):
                    raise ValueError(f"Unexpected dtype {view.get_dtype()} for {name}")
                if name in target_keys:
                    if view.get_dtype() != "BF16":
                        raise ValueError(f"Attention weight must be BF16: {name}")
                    shapes[name] = view.get_shape()
                tensors[name] = handle.get_tensor(name)
                total_size += tensors[name].nbytes
                weight_map[name] = filename
        save_file(tensors, base / filename)
        del tensors
        print(f"Prepared shard {number + 1}/{len(grouped)}", flush=True)
    reduced_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    write_json(base / "model.safetensors.index.json", reduced_index)
    manifest = {
        "layers": layers,
        "rank": rank,
        "alpha": alpha,
        "adapter_std": ADAPTER_STD,
        "synthetic_adapters": True,
        "targets": shapes,
    }
    for label, seed in (("a", 20261121), ("b", 20261122)):
        adapter = output / f"adapter-{label}"
        adapter.mkdir()
        generator = torch.Generator().manual_seed(seed)
        weights = {}
        for name, (out_size, in_size) in shapes.items():
            prefix = "base_model.model." + name.removesuffix(".weight")
            weights[prefix + ".lora_A.weight"] = (
                torch.randn(rank, in_size, generator=generator) * ADAPTER_STD
            ).bfloat16()
            weights[prefix + ".lora_B.weight"] = (
                torch.randn(out_size, rank, generator=generator) * ADAPTER_STD
            ).bfloat16()
        save_file(weights, adapter / "adapter_model.safetensors")
        write_json(
            adapter / "adapter_config.json",
            {
                "base_model_name_or_path": str(base),
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": rank,
                "lora_alpha": alpha,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": list(TARGETS),
                "inference_mode": True,
            },
        )
        merged = output / f"merged-{label}"
        merged.mkdir()
        touched = {weight_map[name] for name in shapes}
        for path in base.iterdir():
            if path.name not in touched:
                (merged / path.name).symlink_to(path.resolve())
        for shard in sorted(touched):
            tensors = load_file(base / shard)
            for name in shapes:
                if weight_map[name] != shard:
                    continue
                prefix = "base_model.model." + name.removesuffix(".weight")
                a = weights[prefix + ".lora_A.weight"].float()
                b = weights[prefix + ".lora_B.weight"].float()
                tensors[name] = (tensors[name].float() + (alpha / rank) * (b @ a)).bfloat16()
            save_file(tensors, merged / shard)
            del tensors
        manifest[label] = {"seed": seed, "adapter_sha256": digest(adapter / "adapter_model.safetensors")}
    manifest["source_config_sha256"] = digest(source / "config.json")
    manifest["source_index_sha256"] = digest(indexes[0])
    manifest["base_config_sha256"] = digest(base / "config.json")
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve(), args.layers, args.rank, args.alpha)
