# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Reject original LoRA failure modes in the independent acceptance checker."""

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).parents[2] / "e2e/models/minimax_m31_lora/reference_checks.py"
_SPEC = importlib.util.spec_from_file_location("m31_reference_checks", _PATH)
checks = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checks)


def record():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    a, b = torch.tensor([[1.0, 0.5]]), torch.tensor([[1.0], [2.0]])
    delta = checks.disk_delta(x, [(a, b)], [(0, 2)], 4, 2)
    base = torch.ones(2, 4)
    return {
        "input": x,
        "base": base,
        "output": base + delta,
        "raw_delta": delta,
        "raw_repeat": delta.clone(),
        "slices": [(0, 2)],
    }, [(a, b)]


@pytest.mark.parametrize("fault", ["missing", "scale", "slice", "indexer", "nan"])
def test_rejects_wrong_increment_or_indexer_corruption(fault):
    values, factors = record()
    if fault == "missing":
        values["raw_delta"].zero_()
    elif fault == "scale":
        values["raw_delta"].mul_(2)
    elif fault == "slice":
        values["raw_delta"] = torch.roll(values["raw_delta"], 1, dims=-1)
    elif fault == "indexer":
        values["output"][:, -1] += 1
    else:
        values["raw_delta"][0, 0] = float("nan")
    assert not checks.inspect_record(values, factors, 2, 1e-6, 1e-5)["passed"]


def test_disk_reference_and_mutation_sensitivity():
    values, factors = record()
    result = checks.inspect_record(values, factors, 2, 1e-6, 1e-5)
    assert result["passed"] and result["signal"]["valid"]
    assert all(result["sensitivity"].values())
    changed_factors = [(factors[0][0] * 3, factors[0][1])]
    assert not checks.inspect_record(values, changed_factors, 2, 1e-6, 1e-5)["passed"]


def test_zero_and_weak_signals_are_invalid_probes():
    assert not checks.signal(torch.zeros(2, 4), 0.01, 0.01)["valid"]
    assert not checks.signal(torch.full((2, 4), 0.001), 0.01, 0.01)["valid"]
    assert not checks.signal(torch.ones(2, 4), 0.01, 0.01, noise=1)["valid"]


def test_bf16_output_rounding_does_not_hide_missing_raw_delta():
    values, factors = record()
    values["base"] = torch.full((2, 4), 4096.0, dtype=torch.bfloat16)
    values["output"] = (values["base"].double() + values["raw_delta"]).bfloat16()
    result = checks.inspect_record(values, factors, 2, 1 / 128, 1 / 128)
    assert result["passed"]
    values["raw_delta"].zero_()
    assert not checks.inspect_record(values, factors, 2, 1 / 128, 1 / 128)["passed"]


def test_summary_requires_complete_evidence_and_rejects_a_missing_decode():
    spec = importlib.util.spec_from_file_location("m31_reference_summary", _PATH.with_name("reference_summary.py"))
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    modules = {
        f"model.language_model.model.layers.{i}.self_attn.{t}": {} for i in range(4) for t in ("qkv_proj", "o_proj")
    }
    labels = [f"{v}-p{i}" for v in ("base", "zero", "a", "b") for i in range(4)] + [
        "isolation-q",
        "isolation-k",
        "isolation-v",
        "lifecycle-loaded",
        "lifecycle-removed",
        "lifecycle-reloaded",
        "after-invalid-base",
        "after-invalid-zero",
    ]
    requests, records = [], []
    for label in labels:
        variant = label.split("-")[0]
        token = {"a": 1, "b": 2}.get(variant, 0)
        requests.append(
            {
                "label": label,
                "token_ids": [token] * 32,
                "logprobs": [{}] * 32,
                "finish_reason": "length",
                "steps": {name: 3 for name in modules},
            }
        )
        records.extend(
            {
                "label": label,
                "module": name,
                "step": step,
                "variant": variant,
                "passed": True,
                "signal": {"valid": True},
                "sensitivity": {"missing": True},
            }
            for name in modules
            for step in range(3)
        )
    layers = {"records": records, "loading": {name: {"base_weight_equal": True} for name in modules}}
    lifecycle = dict.fromkeys(
        (
            "reload_identical",
            "removed_returns_base",
            "invalid_rank_rejected",
            "after_invalid_base",
            "after_invalid_zero",
        ),
        True,
    )
    stable = dict.fromkeys(modules, True)
    model = {
        "coverage_complete": True,
        "cases": [{"label": label, "logprob": {"passed": True}, "greedy_matches": True} for label in labels[:16]],
    }
    assert summary.summarize(layers, requests, modules, lifecycle, stable, model)["status"] == "VERIFIED"
    pending = summary.summarize(layers, requests, modules, lifecycle, stable, None)
    assert pending["status"] == "PARTIAL" and pending["outcome"] == "pending"
    records.pop()
    result = summary.summarize(layers, requests, modules, lifecycle, stable, model)
    assert result["outcome"] == "failed" and not result["checks"]["coverage"]
    layers["records"] = []
    assert not summary.summarize(layers, requests, modules, lifecycle, stable, model)["checks"]["layer_numerics"]
