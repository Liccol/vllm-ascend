# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Test-only controls retaining factorized LoRA, without merged adapter weights."""

import json
from pathlib import Path
from types import MethodType

import torch
from m31_lora_diagnose import LoRADiagnose
from m31_lora_trace import LoRABatchTrace
from torch.nn import functional as F
from vllm.distributed import tensor_model_parallel_all_reduce


def add_delta(layer, x, output, reduce_shrink=False):
    if layer.punica_wrapper.no_lora:
        return output
    shape = output.shape
    x, output = x.reshape(-1, x.shape[-1]).float(), output.reshape(-1, output.shape[-1])
    slots = layer.punica_wrapper.token_lora_indices
    offset = 0
    for a, b, size in zip(layer.lora_a_stacked, layer.lora_b_stacked, layer.output_slices):
        for slot in range(a.shape[0]):
            shrink = x @ a[slot, 0].float().T
            if reduce_shrink and layer.tp_size > 1:
                shrink = tensor_model_parallel_all_reduce(shrink)
            delta = shrink @ b[slot, 0, :size].float().T
            output[:, offset : offset + size] += torch.where((slots == slot).unsqueeze(-1), delta, 0)
        offset += size
    return output.reshape(shape)


class StandardPrecisionProbe(LoRADiagnose):
    def init_worker_extension(self):
        super().init_worker_extension()
        LoRABatchTrace.init_worker_extension(self)
        original = self.load_model

        def load():
            result = original()
            torch.npu.matmul.allow_hf32 = False
            self.install_precision_controls()
            return result

        self.load_model = load

    def install_precision_controls(self):
        from vllm.lora.layers import RowParallelLinearWithLoRA

        marker = Path.cwd() / ".minimax-lora-precision.json"
        for name, layer in self.model_runner.model.named_modules():
            if not hasattr(layer, "lora_a_stacked") or not name.endswith((".qkv_proj", ".o_proj")):
                continue
            row = isinstance(layer, RowParallelLinearWithLoRA)
            original_apply = layer.apply
            original_forward = layer.forward

            def mode():
                return json.loads(marker.read_text())["mode"] if marker.exists() else "original"

            def apply(current, x, bias=None, row=row, original_apply=original_apply):
                if mode() != "fp32_local":
                    return original_apply(x, bias)
                output = F.linear(
                    x.float(), current.base_layer.weight.float(), bias.float() if bias is not None else None
                )
                output = add_delta(current, x, output)
                # Row output remains FP32 until its TP collective finishes.
                return output if row else output.to(x.dtype)

            def forward(current, x, row=row, original_forward=original_forward):
                control = mode()
                if not row or control == "original":
                    return original_forward(x)
                base = current.base_layer
                if not base.input_is_parallel:
                    x = x.chunk(current.tp_size, dim=-1)[current.tp_rank].contiguous()
                bias = base.bias if current.tp_rank == 0 and not base.skip_bias_add else None
                if control == "post_reduce":
                    output = base.quant_method.apply(base, x, bias)
                    if base.reduce_results and current.tp_size > 1:
                        output = tensor_model_parallel_all_reduce(output)
                    output = add_delta(current, x, output.float(), reduce_shrink=True).to(x.dtype)
                elif control == "fp32_local":
                    output = current.apply(x, bias)
                    if base.reduce_results and current.tp_size > 1:
                        output = tensor_model_parallel_all_reduce(output)
                    output = output.to(x.dtype)
                else:
                    raise ValueError(control)
                return (output, base.bias if base.skip_bias_add else None) if base.return_bias else output

            layer.apply = MethodType(apply, layer)
            layer.forward = MethodType(forward, layer)
            print("STANDARD_LOWRANK_PRECISION_CONTROL", self.rank, name, flush=True)
