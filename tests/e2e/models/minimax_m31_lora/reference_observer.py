# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Single-device M3 observer; captures existing kernels without substituting results."""

import hashlib
import inspect
from pathlib import Path

import torch


def signature(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def capture(layer, original, state, name, x, output):
    context = state["context"]
    step = context["steps"].get(name, 0) if context else 3
    if step >= 3:
        return original(x, output)
    context["steps"][name] = step + 1
    count = context["prompt_tokens"] if step == 0 else 1
    values = x.reshape(-1, x.shape[-1])
    if values.shape[0] < count:
        raise ValueError("Observed rows shorter than the single request")
    base = output.reshape(-1, output.shape[-1])[:count].detach().cpu().clone()
    result = original(x, output)
    actual = result.reshape(-1, result.shape[-1])[:count].detach().cpu().clone()
    raw = original(x, torch.zeros_like(output)).reshape(-1, result.shape[-1])[:count].detach().cpu().clone()
    repeat = original(x, torch.zeros_like(output)).reshape(-1, result.shape[-1])[:count].detach().cpu().clone()
    slot = layer.punica_wrapper.token_lora_indices[:count].detach().cpu().clone()
    buffers = {
        "a": [a[0, 0].detach().cpu().clone() for a in layer.lora_a_stacked],
        "b": [b[0, 0].detach().cpu().clone() for b in layer.lora_b_stacked],
    }
    sizes = list(layer.output_slices)
    offsets, start = [], 0
    for size in sizes:
        offsets.append((start, size))
        start += size
    path = Path(state["directory"]) / context["label"] / f"{name}-step{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "module": name,
            "step": step,
            "input": values[:count].detach().cpu().clone(),
            "base": base,
            "output": actual,
            "raw_delta": raw,
            "raw_repeat": repeat,
            "slots": slot,
            "buffers": buffers,
            "slices": offsets,
            "weight_file": state["targets"][name]["weight_file"],
        },
        path,
    )
    context["records"].append(str(path))
    return result


def install(model, directory):
    state = {"directory": directory, "context": None, "targets": {}}
    model._m31_reference = state
    for name, layer in model.named_modules():
        if not name.endswith((".self_attn.qkv_proj", ".self_attn.o_proj")):
            continue
        if not hasattr(layer, "lora_a_stacked") or layer.tp_size != 1:
            raise ValueError(f"Expected an actual single-device LoRA wrapper: {name}")
        weight = layer.base_layer.weight
        path = Path(directory) / "weights" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(weight.detach().cpu(), path)
        state["targets"][name] = {
            "wrapper": type(layer).__name__,
            "source": inspect.getsourcefile(type(layer)),
            "weight_file": str(path),
            "shape": list(weight.shape),
            "weight_sha256": signature(weight),
            "storage": weight.data_ptr(),
            "device": str(weight.device),
        }
        original = layer._apply_lora_to_output

        def apply(x, output, layer=layer, original=original, name=name):
            return capture(layer, original, state, name, x, output)

        layer._apply_lora_to_output = apply
    if not state["targets"]:
        raise ValueError("No M3 attention LoRA wrappers observed")
    return state["targets"]


def begin(model, label, prompt_tokens):
    model._m31_reference["context"] = {"label": label, "prompt_tokens": prompt_tokens, "steps": {}, "records": []}


def end(model):
    result = model._m31_reference["context"]
    model._m31_reference["context"] = None
    return result


def finish(model):
    result = {}
    for name, info in model._m31_reference["targets"].items():
        weight = model.get_submodule(name).base_layer.weight
        result[name] = weight.data_ptr() == info["storage"] and signature(weight) == info["weight_sha256"]
    return result


class ReferenceWorker:
    """Use named worker RPC methods; the client does not enable pickle RPC payloads."""

    def m31_reference_install(self, directory):
        return install(self.model_runner.model, directory)

    def m31_reference_begin(self, label, prompt_tokens):
        begin(self.model_runner.model, label, prompt_tokens)

    def m31_reference_end(self):
        return end(self.model_runner.model)

    def m31_reference_finish(self):
        return finish(self.model_runner.model)
