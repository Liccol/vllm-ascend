# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Disk-factor references for the M3 adaptation protocol, independent of loader buffers."""

import math

import torch


def compare(actual, expected, atol, rtol):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    if actual.shape != expected.shape:
        return {"passed": False, "reason": "shape", "actual": list(actual.shape), "expected": list(expected.shape)}
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = (actual - expected).abs()
    failures = difference > atol + rtol * expected.abs()
    return {
        "passed": finite and not bool(failures.any()),
        "finite": finite,
        "elements": actual.numel(),
        "violations": int(failures.sum()) if finite else actual.numel(),
        "max_absolute": float(difference.max()) if finite else None,
        "relative_l2": float(difference.norm() / expected.norm().clamp_min(1e-30)) if finite else None,
    }


def disk_delta(x, factors, slices, width, scale):
    """A/B come from the original checkpoint; slices are explicit physical output ranges."""
    x = x.detach().cpu().double()
    result = torch.zeros((x.shape[0], width), dtype=torch.float64)
    for (a, b), (start, size) in zip(factors, slices, strict=True):
        if a.shape[1] != x.shape[1] or b.shape != (size, a.shape[0]):
            raise ValueError("Disk factor dimensions do not match the declared slice")
        if start < 0 or start + size > width:
            raise ValueError("Disk factor slice exceeds physical output")
        result[:, start : start + size] += scale * (x @ a.double().T) @ b.double().T
    return result


def signal(delta, atol, rtol, noise=0.0):
    if not bool(torch.isfinite(delta).all()) or not math.isfinite(noise):
        return {"valid": False, "peak": None, "required_margin": None, "repeat_noise": None}
    peak = float(delta.abs().max())
    margin = 10 * (atol + rtol * peak + noise)
    return {"valid": peak > margin, "peak": peak, "required_margin": margin, "repeat_noise": noise}


def inspect_record(record, factors, scale, atol, rtol):
    x, base, output, raw = (record[key].double() for key in ("input", "base", "output", "raw_delta"))
    delta = disk_delta(x, factors, record["slices"], output.shape[1], scale)
    expected = (base + delta).to(record["output"].dtype).double()
    raw_expected = delta.to(record["raw_delta"].dtype).double()
    result = {
        "raw_delta": compare(raw, raw_expected, atol, rtol),
        "output": compare(output, expected, atol, rtol),
        # This metric includes final BF16 output rounding; it is not the isolated branch.
        "applied_delta_vs_ideal": compare(output - base, delta, atol, rtol),
        "signal": signal(delta, atol, rtol, float((raw - record["raw_repeat"].double()).abs().max())),
    }
    end = max(start + size for start, size in record["slices"])
    if end < output.shape[1]:
        result["indexer_unchanged"] = bool(torch.equal(output[:, end:], base[:, end:]))
        result["indexer_delta_zero"] = bool((raw[:, end:] == 0).all())
    for index, (start, size) in enumerate(record["slices"]):
        result[f"slice_{index}_delta"] = compare(
            raw[:, start : start + size], raw_expected[:, start : start + size], atol, rtol
        )
        result[f"slice_{index}_signal"] = signal(delta[:, start : start + size], atol, rtol)
        a, b = factors[index]
        if not bool(b.any()):
            inactive = bool((raw[:, start : start + size] == 0).all()) and torch.equal(
                output[:, start : start + size], base[:, start : start + size]
            )
            result[f"slice_{index}_inactive_exact"] = inactive
        dense = scale * (b[:64].double() @ a.double())
        expanded = record["input"][:4].double() @ dense.T
        result[f"slice_{index}_fp64_crosscheck"] = compare(
            delta[:4, start : start + min(size, 64)], expanded, 1e-10, 1e-10
        )
    result["sensitivity"] = {
        "missing_delta_rejected": not compare(torch.zeros_like(raw), raw_expected, atol, rtol)["passed"],
        "double_scale_rejected": not compare(raw * 2, raw_expected, atol, rtol)["passed"],
        "wrong_slice_rejected": not compare(torch.roll(raw, 1, dims=-1), raw_expected, atol, rtol)["passed"],
    }
    result["passed"] = (
        result["raw_delta"]["passed"]
        and result["output"]["passed"]
        and result.get("indexer_unchanged", True)
        and result.get("indexer_delta_zero", True)
        and all(value for key, value in result.items() if key.endswith("_inactive_exact"))
        and all(value["passed"] for key, value in result.items() if key.endswith("_fp64_crosscheck"))
    )
    return result
