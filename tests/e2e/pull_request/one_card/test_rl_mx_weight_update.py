# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-NPU regression coverage for RL-style Ascend MX weight reloads.

The test deliberately stays below the trainer and transport layers.  It feeds
deterministic BF16 actor-like weights through the same NPU MX primitive used by
verl, restores the checkpoint layout, loads weight and scale as separate
buckets, and rebuilds the inference layout.  The correctness oracle is a fresh
layer loaded from the exact same quantized payload.
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4
from vllm_ascend.quantization.methods.w8a8 import w8a8_mxfp8

DEVICE = torch.device("npu:0")
GROUP_SIZE = 32
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 128
OUTPUT_SIZE = 128
TOKENS = 8

pytestmark = pytest.mark.skipif(torch.npu.device_count() < 1, reason="RL MX weight reload tests require one NPU")


@dataclass(frozen=True)
class MXCase:
    id: str
    linear_cls: type
    moe_cls: type
    dst_type: torch.dtype
    packed: bool
    transformed_flag: str


MX_CASES = (
    MXCase(
        id="mxfp8",
        linear_cls=w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod,
        moe_cls=w8a8_mxfp8.AscendW8A8MXFP8DynamicFusedMoEMethod,
        dst_type=torch_npu.float8_e4m3fn,
        packed=False,
        transformed_flag="_mxfp8_transformed",
    ),
    MXCase(
        id="mxfp4",
        linear_cls=w4a8_mxfp4.AscendW4A8MXFPDynamicLinearMethod,
        moe_cls=w4a8_mxfp4.AscendW4A8MXFPDynamicFusedMoEMethod,
        dst_type=torch_npu.float4_e2m1fn_x2,
        packed=True,
        transformed_flag="_mxfp4_transformed",
    ),
)


def _make_method(method_cls):
    """Construct a scheme without depending on global engine configuration."""
    method = object.__new__(method_cls)
    method.group_size = GROUP_SIZE
    method.dynamic_eplb = False
    if isinstance(method, w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod):
        method.dynamic_mx_quant_scale_alg = 0
    return method


def _actor_weight(shape: tuple[int, ...], update: int) -> torch.Tensor:
    """Return deterministic BF16 values whose MX mantissas and scales change."""
    generator = torch.Generator(device="cpu").manual_seed(20261013 + update * 97 + len(shape))
    weight = torch.randn(shape, generator=generator, dtype=torch.float32)
    ramp = torch.linspace(-1.5, 2.0, weight.numel(), dtype=torch.float32).reshape(shape)
    weight = weight * (0.65 + 0.2 * update) + torch.sin(ramp * (update + 1))
    return weight.to(torch.bfloat16).to(DEVICE)


def _quantize(weight: torch.Tensor, case: MXCase) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-1, dst_type=case.dst_type)
    scale = scale.flatten(-2, -1).squeeze(-1)
    if case.packed:
        quantized = quantized.view(torch.uint8)
    return quantized, scale


def _assert_same_tensor(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual.float().cpu(), expected.float().cpu(), rtol=0, atol=0)


def _load_parameter(parameter: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Use the same per-parameter loader selected by vLLM model loading."""
    weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
    weight_loader(parameter, loaded_weight)


def _assert_payload_changed(
    previous: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    weight_names: tuple[str, ...],
) -> None:
    for name in weight_names:
        assert not torch.equal(previous[name].view(torch.uint8).cpu(), current[name].view(torch.uint8).cpu()), name
        scale_name = f"{name}_scale"
        assert not torch.equal(
            previous[scale_name].view(torch.uint8).cpu(), current[scale_name].view(torch.uint8).cpu()
        ), scale_name


def _dense_payload(case: MXCase, update: int) -> dict[str, torch.Tensor]:
    weight = _actor_weight((OUTPUT_SIZE, HIDDEN_SIZE), update)
    quantized, scale = _quantize(weight, case)
    return {"weight": quantized, "weight_scale": scale}


def _make_dense_layer(method) -> nn.Module:
    layer = nn.Module()
    tensors = {
        **method.get_weight(HIDDEN_SIZE, OUTPUT_SIZE, torch.bfloat16),
        **method.get_pergroup_param(HIDDEN_SIZE, OUTPUT_SIZE, torch.bfloat16),
    }
    for name, tensor in tensors.items():
        layer.register_parameter(name, nn.Parameter(tensor.to(DEVICE), requires_grad=False))
    return layer


def _load_dense_buckets(layer: nn.Module, payload: dict[str, torch.Tensor]) -> None:
    """Model the two-bucket order used by an RL weight transfer."""
    _load_parameter(layer.weight, payload["weight"])
    assert not getattr(layer, "_mxfp8_transformed", False)
    assert not getattr(layer, "_mxfp4_transformed", False)
    _load_parameter(layer.weight_scale, payload["weight_scale"])


def _fresh_dense(method, payload: dict[str, torch.Tensor]) -> nn.Module:
    layer = _make_dense_layer(method)
    _load_dense_buckets(layer, payload)
    method.process_weights_after_loading(layer)
    return layer


def _reload_dense(method, layer: nn.Module, payload: dict[str, torch.Tensor]) -> None:
    method.restore_weights_for_rl_loading(layer)
    _load_dense_buckets(layer, payload)
    method.process_weights_after_loading(layer)


def _dense_output(method, layer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return method.apply(layer, inputs.clone())


def _moe_payload(case: MXCase, update: int) -> dict[str, torch.Tensor]:
    sources = {
        "w13_weight": _actor_weight((2, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), update),
        "w2_weight": _actor_weight((2, HIDDEN_SIZE, INTERMEDIATE_SIZE), update + 11),
    }
    payload = {}
    for name, source in sources.items():
        quantized, scale = _quantize(source, case)
        payload[name] = quantized
        payload[f"{name}_scale"] = scale
    return payload


def _make_moe_layer(method) -> nn.Module:
    layer = nn.Module()
    tensors = {
        **method.get_weight(2, INTERMEDIATE_SIZE, HIDDEN_SIZE, torch.bfloat16),
        **method.get_dynamic_quant_param(2, INTERMEDIATE_SIZE, HIDDEN_SIZE, torch.bfloat16),
    }
    for name, tensor in tensors.items():
        layer.register_parameter(name, nn.Parameter(tensor.to(DEVICE), requires_grad=False))
    layer.activation = MoEActivation.SILU
    return layer


def _load_moe_buckets(layer: nn.Module, payload: dict[str, torch.Tensor]) -> None:
    for name in ("w13_weight", "w2_weight"):
        _load_parameter(getattr(layer, name), payload[name])
    assert not getattr(layer, "_mxfp8_transformed", False)
    assert not getattr(layer, "_mxfp4_transformed", False)
    for name in ("w13_weight_scale", "w2_weight_scale"):
        _load_parameter(getattr(layer, name), payload[name])


def _without_megamoe(case: MXCase):
    module = w4a8_mxfp4 if case.packed else w8a8_mxfp8
    return patch.object(module, "use_cann_megamoe", return_value=False)


def _fresh_moe(case: MXCase, method, payload: dict[str, torch.Tensor]) -> nn.Module:
    layer = _make_moe_layer(method)
    _load_moe_buckets(layer, payload)
    with _without_megamoe(case):
        method.process_weights_after_loading(layer)
    return layer


def _reload_moe(case: MXCase, method, layer: nn.Module, payload: dict[str, torch.Tensor]) -> None:
    method.restore_weights_for_rl_loading(layer)
    _load_moe_buckets(layer, payload)
    with _without_megamoe(case):
        method.process_weights_after_loading(layer)


def _moe_output(method, layer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    group_list = torch.tensor([TOKENS // 2, TOKENS // 2], dtype=torch.int64, device=DEVICE)
    mlp_input = MoEMlpComputeInput(
        hidden_states=inputs.clone(),
        group_list=group_list,
        group_list_type=1,
        dynamic_scale=None,
        topk_scales=None,
        weights=MoEWeights(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        ),
        quant=MoEQuantParams(quant_type=method.quant_type),
        fusion=True,
        layer=layer,
        activation=MoEActivation.SILU,
    )
    hidden_states, activation_scale = method.apply_gmm1_act_quant(mlp_input)
    return method.apply_gmm2(mlp_input, hidden_states, activation_scale)


@pytest.mark.parametrize("case", MX_CASES, ids=lambda case: case.id)
def test_dense_mx_weights_match_fresh_load_across_two_rl_updates(case: MXCase):
    torch.npu.set_device(DEVICE)
    torch.manual_seed(20261013)
    method = _make_method(case.linear_cls)
    payloads = [_dense_payload(case, update) for update in range(3)]
    _assert_payload_changed(payloads[0], payloads[1], ("weight",))
    _assert_payload_changed(payloads[1], payloads[2], ("weight",))
    inputs = _actor_weight((TOKENS, HIDDEN_SIZE), 31)

    live = _fresh_dense(method, payloads[0])
    initial_output = _dense_output(method, live, inputs)
    stable_ptrs = None
    if case.id == "mxfp8":
        stable_ptrs = (live.weight.data_ptr(), live.weight_scale.data_ptr())

    for payload in payloads[1:]:
        _reload_dense(method, live, payload)
        reference = _fresh_dense(method, payload)
        _assert_same_tensor(live.weight, reference.weight)
        _assert_same_tensor(live.weight_scale, reference.weight_scale)
        torch.testing.assert_close(
            _dense_output(method, live, inputs).cpu(),
            _dense_output(method, reference, inputs).cpu(),
            rtol=0,
            atol=0,
        )
        assert getattr(live, case.transformed_flag)
        if stable_ptrs is not None:
            assert (live.weight.data_ptr(), live.weight_scale.data_ptr()) == stable_ptrs

    final_output = _dense_output(method, live, inputs)
    assert not torch.equal(initial_output.cpu(), final_output.cpu())
    assert torch.isfinite(final_output).all()


@pytest.mark.parametrize("case", MX_CASES, ids=lambda case: case.id)
def test_moe_mx_weights_match_fresh_load_across_two_rl_updates(case: MXCase):
    torch.npu.set_device(DEVICE)
    torch.manual_seed(20261013)
    method = _make_method(case.moe_cls)
    payloads = [_moe_payload(case, update) for update in range(3)]
    weight_names = ("w13_weight", "w2_weight")
    _assert_payload_changed(payloads[0], payloads[1], weight_names)
    _assert_payload_changed(payloads[1], payloads[2], weight_names)
    inputs = _actor_weight((TOKENS, HIDDEN_SIZE), 41)

    live = _fresh_moe(case, method, payloads[0])
    initial_output = _moe_output(method, live, inputs)
    stable_ptrs = None
    if case.id == "mxfp8":
        stable_ptrs = tuple(
            getattr(live, name).data_ptr() for name in (*weight_names, *(f"{n}_scale" for n in weight_names))
        )

    for payload in payloads[1:]:
        _reload_moe(case, method, live, payload)
        reference = _fresh_moe(case, method, payload)
        for name in (*weight_names, *(f"{n}_scale" for n in weight_names)):
            _assert_same_tensor(getattr(live, name), getattr(reference, name))
        torch.testing.assert_close(
            _moe_output(method, live, inputs).cpu(),
            _moe_output(method, reference, inputs).cpu(),
            rtol=0,
            atol=0,
        )
        assert getattr(live, case.transformed_flag)
        if stable_ptrs is not None:
            current_ptrs = tuple(
                getattr(live, name).data_ptr() for name in (*weight_names, *(f"{n}_scale" for n in weight_names))
            )
            assert current_ptrs == stable_ptrs

    final_output = _moe_output(method, live, inputs)
    assert not torch.equal(initial_output.cpu(), final_output.cpu())
    assert torch.isfinite(final_output).all()


def test_mixed_mxfp8_and_bf16_weights_match_fresh_load():
    torch.npu.set_device(DEVICE)
    method = _make_method(w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod)
    inputs = _actor_weight((TOKENS, HIDDEN_SIZE), 51)
    initial_mx = _dense_payload(MX_CASES[0], 0)
    updated_mx = _dense_payload(MX_CASES[0], 1)
    initial_bf16 = _actor_weight((OUTPUT_SIZE, HIDDEN_SIZE), 61)
    updated_bf16 = _actor_weight((OUTPUT_SIZE, HIDDEN_SIZE), 62)

    live = _fresh_dense(method, initial_mx)
    live_residual = nn.Parameter(initial_bf16.clone(), requires_grad=False)
    initial_output = _dense_output(method, live, inputs) + F.linear(inputs, live_residual)

    method.restore_weights_for_rl_loading(live)
    _load_parameter(live.weight, updated_mx["weight"])
    _load_parameter(live_residual, updated_bf16)
    _load_parameter(live.weight_scale, updated_mx["weight_scale"])
    method.process_weights_after_loading(live)

    reference = _fresh_dense(method, updated_mx)
    reference_residual = nn.Parameter(updated_bf16.clone(), requires_grad=False)
    hot_output = _dense_output(method, live, inputs) + F.linear(inputs, live_residual)
    fresh_output = _dense_output(method, reference, inputs) + F.linear(inputs, reference_residual)

    _assert_same_tensor(live.weight, reference.weight)
    _assert_same_tensor(live.weight_scale, reference.weight_scale)
    _assert_same_tensor(live_residual, reference_residual)
    torch.testing.assert_close(hot_output.cpu(), fresh_output.cpu(), rtol=0, atol=0)
    assert not torch.equal(initial_output.cpu(), hot_output.cpu())
    assert torch.isfinite(hot_output).all()
