# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Finalize the independent protocol; absent or empty evidence cannot verify a model."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(layers, requests, inventory, lifecycle, stability, model):
    expected_labels = {f"{v}-p{i}" for v in ("base", "zero", "a", "b") for i in range(4)} | {
        "isolation-q",
        "isolation-k",
        "isolation-v",
        "lifecycle-loaded",
        "lifecycle-removed",
        "lifecycle-reloaded",
        "after-invalid-base",
        "after-invalid-zero",
    }
    expected_modules = {
        f"model.language_model.model.layers.{i}.self_attn.{target}"
        for i in range(4)
        for target in ("qkv_proj", "o_proj")
    }
    expected_records = Counter(
        (label, module, step) for label in expected_labels for module in expected_modules for step in range(3)
    )
    records = layers["records"]
    actual_records = Counter((r["label"], r["module"], r["step"]) for r in records)
    by_label = {r["label"]: r for r in requests}
    coverage = (
        set(inventory) == expected_modules
        and len(requests) == len(expected_labels)
        and set(by_label) == expected_labels
        and actual_records == expected_records
        and all(len(r["token_ids"]) == 32 and len(r["logprobs"]) == 32 for r in requests)
        and all(set(r["steps"]) == expected_modules and set(r["steps"].values()) == {3} for r in requests)
    )
    same = lambda a, b: all(a[k] == b[k] for k in ("token_ids", "finish_reason", "logprobs"))
    zero_equal = all(
        f"base-p{i}" in by_label and f"zero-p{i}" in by_label and same(by_label[f"base-p{i}"], by_label[f"zero-p{i}"])
        for i in range(4)
    )
    distinct = all(
        any(
            f"{a}-p{i}" in by_label
            and f"{b}-p{i}" in by_label
            and by_label[f"{a}-p{i}"]["token_ids"] != by_label[f"{b}-p{i}"]["token_ids"]
            for i in range(4)
        )
        for a, b in (("base", "a"), ("base", "b"), ("a", "b"))
    )
    signal_rows = [r for r in records if r["variant"] in {"a", "b"}]
    signal = bool(signal_rows) and all(
        r["signal"]["valid"]
        and all(r["sensitivity"].values())
        and all(value["valid"] for key, value in r.items() if key.startswith("slice_") and key.endswith("_signal"))
        for r in signal_rows
    )
    expected_model = {f"{v}-p{i}" for v in ("base", "zero", "a", "b") for i in range(4)}
    model_complete = (
        model is not None
        and model.get("coverage_complete")
        and (len(model["cases"]) == 16 and {r["label"] for r in model["cases"]} == expected_model)
    )
    checks = {
        "coverage": coverage,
        "layer_numerics": bool(records) and all(r["passed"] for r in records),
        "signal_and_sensitivity": signal,
        "disk_base_weights": set(layers["loading"]) == expected_modules
        and all(r["base_weight_equal"] for r in layers["loading"].values()),
        "base_weights_stable": set(stability) == expected_modules and all(stability.values()),
        "zero_equals_base": zero_equal,
        "fixtures_distinct": distinct,
        "lifecycle": all(
            lifecycle.get(k) is True
            for k in (
                "reload_identical",
                "removed_returns_base",
                "invalid_rank_rejected",
                "after_invalid_base",
                "after_invalid_zero",
            )
        ),
        "model_reference": None
        if model is None
        else bool(model_complete and all(r["logprob"]["passed"] and r["greedy_matches"] for r in model["cases"])),
    }
    verified = all(value is True for value in checks.values())
    return {
        "status": "VERIFIED" if verified else "PARTIAL",
        "outcome": "passed" if verified else "failed" if False in checks.values() else "pending",
        "scope": "four real BF16 layers; TP1/DP1; single adapter; eager; Q/K/V/O",
        "checks": checks,
        "layer_records_passed": sum(r["passed"] for r in records),
        "layer_records_total": len(records),
        "legacy_strict_results": "unchanged; this protocol does not replace merged or cross-TP acceptance",
    }


def finalize(output):
    read = lambda name: json.loads((output / (name + ".json")).read_text())
    model = read("model-reference") if (output / "model-reference.json").exists() else None
    result = summarize(
        read("layer-results"), read("requests"), read("inventory"), read("lifecycle"), read("base-stability"), model
    )
    (output / "final-summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    result = finalize(parser.parse_args().output)
    print(json.dumps(result, indent=2))
    raise SystemExit({"passed": 0, "failed": 1, "pending": 2}[result["outcome"]])
