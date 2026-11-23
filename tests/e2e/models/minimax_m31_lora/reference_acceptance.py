# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""M3 BF16 adaptation checks inspired by MindSpeed-Agent a86e853's LoRA protocol."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from reference_checks import compare, disk_delta, inspect_record
from reference_summary import finalize
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from verify import PROMPTS


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(weights, out):
    config = json.loads((weights / "adapter-a/adapter_config.json").read_text())
    a = load_file(weights / "adapter-a/adapter_model.safetensors")
    paths = {"a": weights / "adapter-a", "b": weights / "adapter-b"}
    for variant in ("zero", "q", "k", "v", "invalid"):
        path = out / "adapters" / variant
        path.mkdir(parents=True)
        tensors = {name: value.clone() for name, value in a.items()}
        for name, value in tensors.items():
            if name.endswith(".lora_B.weight") and (
                variant == "zero" or variant in {"q", "k", "v"} and f".{variant}_proj." not in name
            ):
                value.zero_()
        current = dict(config)
        if variant == "invalid":
            current["r"] = config["r"] + 1
        save_file(tensors, path / "adapter_model.safetensors")
        write(path / "adapter_config.json", current)
        paths[variant] = path
    return config, paths


def checkpoint_weight(base, key, index):
    with safe_open(base / index[key], framework="pt") as handle:
        return handle.get_tensor(key)


def analyze(out, weights, config, paths, inventory, requests, atol, rtol):
    index = json.loads((weights / "base/model.safetensors.index.json").read_text())["weight_map"]
    factors_by_variant = {name: load_file(path / "adapter_model.safetensors") for name, path in paths.items()}
    scale = config["lora_alpha"] / config["r"]
    results, loading = [], {}
    for name, info in inventory.items():
        prefix = name.removeprefix("model.").rsplit(".", 1)[0]
        targets = ("q_proj", "k_proj", "v_proj") if name.endswith("qkv_proj") else ("o_proj",)
        base_targets = (*targets, "index_q_proj", "index_k_proj") if len(targets) == 3 else targets
        weight = torch.load(info["weight_file"], map_location="cpu", weights_only=True)
        expected = torch.cat(
            [checkpoint_weight(weights / "base", prefix + "." + target + ".weight", index) for target in base_targets],
            dim=0,
        )
        loading[name] = {"base_weight_equal": torch.equal(weight, expected), "wrapper": info["wrapper"]}
        weight64 = weight.double()
        for request in requests:
            variant = request["variant"]
            factors = []
            tensors = factors_by_variant.get(variant, factors_by_variant["zero"])
            for target in targets:
                key = "base_model.model." + prefix + "." + target
                factors.append((tensors[key + ".lora_A.weight"], tensors[key + ".lora_B.weight"]))
            for path in request["records"]:
                if not Path(path).name.startswith(name + "-step"):
                    continue
                record = torch.load(path, map_location="cpu", weights_only=True)
                row = inspect_record(record, factors, scale, atol, rtol)
                row.update(module=name, label=request["label"], variant=variant, step=record["step"])
                row["slots_correct"] = bool((record["slots"] == (-1 if variant == "base" else 0)).all())
                buffer_ok = True
                if variant != "base":
                    for slot, (a, b) in enumerate(factors):
                        actual_a, actual_b = record["buffers"]["a"][slot], record["buffers"]["b"][slot]
                        buffer_ok &= torch.equal(actual_a[: a.shape[0]], a)
                        buffer_ok &= torch.equal(actual_b[: b.shape[0], : b.shape[1]], b * scale)
                        buffer_ok &= not bool(actual_a[a.shape[0] :].any())
                        buffer_ok &= not bool(actual_b[:, b.shape[1] :].any())
                        buffer_ok &= not bool(actual_b[b.shape[0] :].any())
                row["buffers_correct"] = buffer_ok
                full_reference = record["input"].double() @ weight64.T
                delta = disk_delta(record["input"], factors, record["slices"], weight.shape[0], scale)
                row["fp64_base_plus_delta"] = compare(record["output"], full_reference + delta, atol, rtol)
                row["passed"] &= row["slots_correct"] and buffer_ok and row["fp64_base_plus_delta"]["passed"]
                results.append(row)
        print("ANALYZED", name, flush=True)
    report = {
        "loading": loading,
        "records": results,
        "records_passed": sum(row["passed"] for row in results),
        "records_total": len(results),
    }
    write(out / "layer-results.json", report)
    return report


def run(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    config, paths = prepare(args.weights, args.output)
    atol = rtol = 1 / 128  # One BF16 epsilon, frozen before observing this run.
    contract = {
        "protocol_source": "MindSpeed-Agent a86e85307e87888ad6ab0ac41e4b5925ca45b65c",
        "scope": "M3 first four real BF16 layers, Q/K/V/O, one device and one active adapter",
        "atol": atol,
        "rtol": rtol,
        "capture_steps": 3,
        "generated_tokens": 32,
        "weights": str(args.weights),
        "adapters": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
        "checks": ["disk_buffers", "raw_delta", "output", "zero", "slice_isolation", "lifecycle", "invalid_rank"],
        "legacy_strict_cases": "unchanged and reported separately",
    }
    write(args.output / "contract.json", contract)
    llm = LLM(
        model=str(args.weights / "base"),
        dtype="bfloat16",
        language_model_only=True,
        enable_lora=True,
        max_loras=1,
        max_cpu_loras=1,
        max_lora_rank=config["r"],
        lora_dtype="bfloat16",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        enforce_eager=True,
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.45,
        block_size=128,
        generation_config="vllm",
        seed=0,
        worker_extension_cls="reference_observer.ReferenceWorker",
    )
    inventory = llm.collective_rpc("m31_reference_install", args=(str(args.output / "observations"),))[0]
    write(args.output / "inventory.json", inventory)
    tokenizer = AutoTokenizer.from_pretrained(args.weights / "base", trust_remote_code=False, local_files_only=True)
    prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in PROMPTS]
    requests, operations = [], []

    def generate(variant, prompt_id, label):
        manager = llm.llm_engine
        for active in manager.list_loras():
            manager.remove_lora(active)
            retained = sorted(manager.list_loras())
            if active in retained:
                raise RuntimeError("Adapter removal was not observed")
            operations.append(
                {
                    "action": "remove",
                    "id": active,
                    "retained": retained,
                }
            )
        lora = None
        if variant != "base":
            lora = LoRARequest(variant, 1, str(paths[variant]))
            manager.add_lora(lora)
            retained = sorted(manager.list_loras())
            if 1 not in retained:
                raise RuntimeError("Adapter load was not observed")
            operations.append(
                {
                    "action": "load",
                    "variant": variant,
                    "retained": retained,
                }
            )
        llm.collective_rpc("m31_reference_begin", args=(label, len(prompts[prompt_id])))
        result = llm.generate(
            [{"prompt_token_ids": prompts[prompt_id]}],
            SamplingParams(temperature=0, seed=0, max_tokens=32, ignore_eos=True, logprobs=5),
            lora_request=lora,
            use_tqdm=False,
        )[0]
        observed = llm.collective_rpc("m31_reference_end")[0]
        output = result.outputs[0]
        row = {
            "label": label,
            "variant": variant,
            "prompt_id": prompt_id,
            "prompt_token_ids": prompts[prompt_id],
            "token_ids": list(output.token_ids),
            "finish_reason": output.finish_reason,
            "logprobs": [{str(k): float(v.logprob) for k, v in step.items()} for step in output.logprobs],
            "records": observed["records"],
            "steps": observed["steps"],
        }
        requests.append(row)
        write(args.output / "requests.json", requests)
        print("REQUEST", label, len(output.token_ids), flush=True)
        return row

    try:
        for variant in ("base", "zero", "a", "b"):
            for prompt_id in range(4):
                generate(variant, prompt_id, f"{variant}-p{prompt_id}")
        for variant in ("q", "k", "v"):
            generate(variant, 0, f"isolation-{variant}")
        first = generate("a", 0, "lifecycle-loaded")
        base = generate("base", 0, "lifecycle-removed")
        reloaded = generate("a", 0, "lifecycle-reloaded")
        llm.llm_engine.remove_lora(1)
        rejection = None
        try:
            llm.llm_engine.add_lora(LoRARequest("invalid", 99, str(paths["invalid"])))
        except Exception as error:
            rejection = str(error)
        restored_base = generate("base", 0, "after-invalid-base")
        restored_zero = generate("zero", 0, "after-invalid-zero")
        base_ref = next(row for row in requests if row["label"] == "base-p0")
        same = lambda a, b: all(a[key] == b[key] for key in ("token_ids", "finish_reason", "logprobs"))
        lifecycle = {
            "reload_identical": same(first, reloaded),
            "removed_returns_base": same(base, base_ref),
            "invalid_rank_rejected": bool(
                rejection and re.search(r"LoRA rank .*greater than max_lora_rank", rejection)
            ),
            "rejection": rejection,
            "after_invalid_base": same(restored_base, base_ref),
            "after_invalid_zero": same(restored_zero, base_ref),
            "operations": operations,
        }
        write(args.output / "lifecycle.json", lifecycle)
        weights_stable = llm.collective_rpc("m31_reference_finish")[0]
        write(args.output / "base-stability.json", weights_stable)
    finally:
        llm.llm_engine.engine_core.shutdown()
    analyze(args.output, args.weights, config, paths, inventory, requests, atol, rtol)
    summary = finalize(args.output)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = run(parser.parse_args())
    raise SystemExit({"passed": 0, "failed": 1, "pending": 2}[result["outcome"]])
