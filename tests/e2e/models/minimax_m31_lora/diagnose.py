# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Test-only snapshots and independent arithmetic checks for a serial request."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from vllm_ascend.utils import maybe_trans_nz


def error(actual, expected):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    difference = actual - expected
    return {
        "equal": torch.equal(actual, expected),
        "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-30)).item(),
        "max_absolute": difference.abs().max().item(),
        "unequal_fraction": (actual != expected).float().mean().item(),
    }


class LoRADiagnose:
    """Enable only on an idle correctness service, never for performance."""

    def init_worker_extension(self):
        original = self.load_model

        def load():
            result = original()
            self.install_lora_diagnostics()
            return result

        self.load_model = load

    def install_lora_diagnostics(self):
        marker = Path.cwd() / ".minimax-lora-diagnose.json"
        rank = self.rank
        dp = self.vllm_config.parallel_config.data_parallel_rank
        model = self.model_runner.model
        seen = set()

        def settings():
            if getattr(self.model_runner.model_state, "kvpp_is_dummy_run", False):
                return None
            if not marker.exists():
                return None
            config = json.loads(marker.read_text())
            if config.get("role") != self.vllm_config.model_config.served_model_name:
                return None
            return config

        def save_once(name, values):
            config = settings()
            if config is None or not config.get("snapshots", False):
                return
            key = (config["label"], name)
            if key in seen:
                return
            seen.add(key)
            directory = Path(config["directory"]) / config["label"]
            directory.mkdir(parents=True, exist_ok=True)
            tensors = {
                key: value.detach().cpu().clone() for key, value in values.items() if isinstance(value, torch.Tensor)
            }
            torch.save(tensors, directory / f"dp{dp}-rank{rank}-{name}.pt")

        def write(row):
            config = settings()
            if config is None:
                return
            directory = Path(config["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            row.update(label=config["label"], dp_rank=dp, rank=rank)
            with (directory / f"arithmetic-dp{dp}-rank{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

        def hook(name):
            def record(module, args, output):
                values = {f"input{i}": value for i, value in enumerate(args)}
                if isinstance(output, torch.Tensor):
                    values["output"] = output
                elif isinstance(output, tuple):
                    values.update({f"output{i}": value for i, value in enumerate(output)})
                save_once(name, values)

            return record

        for name, module in model.named_modules():
            if ".base_layer" in name:
                continue
            if name.endswith((".qkv_proj", ".o_proj", ".gate", ".norm", ".embed_tokens")) or (
                ".layers." in name and name.rsplit(".", 1)[-1].isdigit()
            ):
                module.register_forward_hook(hook(name))
            if name.endswith((".qkv_proj", ".o_proj")) and hasattr(module, "lora_a_stacked"):
                self.wrap_lora_arithmetic(name, module, settings, write, seen)

        original_logits = model.compute_logits

        def logits(hidden, *args, **kwargs):
            result = original_logits(hidden, *args, **kwargs)
            save_once("logits", {"hidden": hidden, "logits": result})
            return result

        model.compute_logits = logits

    def wrap_lora_arithmetic(self, name, layer, settings, write, seen):
        original = layer._apply_lora_to_output
        merged_cache = {}

        def apply(x, output):
            config = settings()
            if config is None:
                return original(x, output)
            if config.get("mode") == "merged-reference":
                key = (config["label"], "merged-reference", name)
                if key not in seen:
                    seen.add(key)
                    write({"module": name, "test_only_dense_merge": True})
                return self.merged_lora_reference(layer, x, output, config["adapter"], merged_cache)
            if config.get("mode") == "unmerged-reference":
                key = (config["label"], "unmerged-reference", name)
                if key not in seen:
                    seen.add(key)
                    write({"module": name, "test_only_fp64_delta": True})
                return self.unmerged_lora_reference(layer, x, output)
            key = (config["label"], "arithmetic", name)
            if key in seen:
                return original(x, output)
            seen.add(key)
            checks = self.check_lora_weights(name, layer, Path(config["weights"]))
            before = output.detach().cpu().float()
            x_cpu = x.detach().cpu().float().reshape(-1, x.shape[-1])
            indices = layer.punica_wrapper._token_lora_indices[: x_cpu.shape[0]].cpu().tolist()
            expected = before.reshape(-1, before.shape[-1]).clone()
            offset = 0
            for a, b, size in zip(layer.lora_a_stacked, layer.lora_b_stacked, layer.output_slices):
                a, b = a.detach().cpu().float(), b.detach().cpu().float()
                for slot in sorted(set(indices)):
                    if slot < 0:
                        continue
                    rows = [i for i, value in enumerate(indices) if value == slot]
                    delta = (x_cpu[rows] @ a[slot, 0].T) @ b[slot, 0, :size].T
                    expected[rows, offset : offset + size] += delta
                offset += size
            result = original(x, output)
            expected = expected.reshape(before.shape).to(result.dtype)
            write({"module": name, "weight_checks": checks, "slots": indices, "oracle": error(result, expected)})
            return result

        layer._apply_lora_to_output = apply

    def unmerged_lora_reference(self, layer, x, output):
        """Replace only the low-rank kernel with independent CPU FP64 arithmetic."""
        values = x.detach().cpu().double().reshape(-1, x.shape[-1])
        result = output.detach().cpu().double().reshape(-1, output.shape[-1])
        indices = layer.punica_wrapper._token_lora_indices[: len(values)].cpu().tolist()
        offset = 0
        for a, b, size in zip(layer.lora_a_stacked, layer.lora_b_stacked, layer.output_slices):
            a, b = a.detach().cpu().double(), b.detach().cpu().double()
            for slot in sorted(set(indices)):
                if slot < 0:
                    continue
                rows = [i for i, value in enumerate(indices) if value == slot]
                result[rows, offset : offset + size] += (values[rows] @ a[slot, 0].T) @ b[slot, 0, :size].T
            offset += size
        return result.reshape(output.shape).to(output.device, output.dtype)

    def merged_lora_reference(self, layer, x, output, adapter, cache):
        """Causal control only: this replaces low-rank inference with a dense GEMM."""
        indices = layer.punica_wrapper._token_lora_indices[: x.reshape(-1, x.shape[-1]).shape[0]].cpu().tolist()
        slots = set(indices)
        if slots == {-1}:
            return output
        if len(slots) != 1 or min(slots) < 0:
            raise ValueError("The merged reference requires one serial adapter request")
        slot = indices[0]
        key = (adapter, slot)
        if key not in cache:
            merged = layer.base_layer.weight.detach().cpu().clone()
            offset = 0
            for a, b, size in zip(layer.lora_a_stacked, layer.lora_b_stacked, layer.output_slices):
                a, b = a[slot, 0].cpu().float(), b[slot, 0, :size].cpu().float()
                merged[offset : offset + size] = (merged[offset : offset + size].float() + b @ a).bfloat16()
                offset += size
            cache[key] = maybe_trans_nz(merged.to(x.device))
        return torch.nn.functional.linear(x, cache[key])

    def check_lora_weights(self, name, layer, weights):
        """Check checkpoint slices directly, independently of wrapper slice helpers."""
        adapter_files = [load_file(weights / f"adapter-{suffix}/adapter_model.safetensors") for suffix in ("a", "b")]
        base_index = json.loads((weights / "base/model.safetensors.index.json").read_text())["weight_map"]

        def read(key):
            with safe_open(weights / "base" / base_index[key], framework="pt") as handle:
                return handle.get_tensor(key)

        targets = ("q_proj", "k_proj", "v_proj") if name.endswith("qkv_proj") else ("o_proj",)
        offset = 0
        checks = []
        for i, (target, size) in enumerate(zip(targets, layer.output_slices)):
            source_name = name.removeprefix("model.").rsplit(".", 1)[0] + "." + target
            key = source_name + ".weight"
            base = read(key)
            if target == "o_proj":
                start = layer.tp_rank * layer.input_size
                expected_base = base[:, start : start + layer.input_size]
            else:
                replicas = layer.base_layer.num_kv_head_replicas if target != "q_proj" else 1
                start = (layer.tp_rank // replicas) * size
                expected_base = base[start : start + size]
            actual_base = layer.base_layer.weight.detach().cpu()
            if len(targets) > 1:
                actual_base = actual_base[offset : offset + size]
            row = {"target": target, "base_equal": torch.equal(actual_base, expected_base), "adapters": []}
            for slot in range(len(layer.lora_a_stacked[i])):
                actual_a = layer.lora_a_stacked[i][slot, 0].detach().cpu()
                actual_b = layer.lora_b_stacked[i][slot, 0, :size].detach().cpu()
                matches = []
                for suffix, tensors in zip(("a", "b"), adapter_files):
                    prefix = "base_model.model." + source_name
                    a, b = tensors[prefix + ".lora_A.weight"], tensors[prefix + ".lora_B.weight"]
                    config = json.loads((weights / f"adapter-{suffix}/adapter_config.json").read_text())
                    b = b * (config["lora_alpha"] / config["r"])
                    if target == "o_proj":
                        a = a[:, start : start + layer.input_size]
                    else:
                        b = b[start : start + size]
                    if torch.equal(actual_a, a) and torch.equal(actual_b, b):
                        matches.append(suffix)
                row["adapters"].append({"slot": slot, "matches": matches, "zero": not bool(actual_a.any())})
            checks.append(row)
            offset += size
        return checks
