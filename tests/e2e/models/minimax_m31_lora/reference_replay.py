# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Diagnostic CPU replay of single-slot O's BF16 intermediate, never an acceptance gate."""

import argparse
import json
from pathlib import Path

import torch
from reference_checks import compare
from safetensors.torch import load_file


def run(weights, output):
    torch.set_num_threads(8)
    contract = json.loads((output / "contract.json").read_text())
    requests = json.loads((output / "requests.json").read_text())
    factors = {v: load_file(weights / f"adapter-{v}/adapter_model.safetensors") for v in ("a", "b")}
    results = []
    for request in requests:
        if request["variant"] not in factors:
            continue
        for path in request["records"]:
            if ".o_proj-" not in path:
                continue
            record = torch.load(path, map_location="cpu", weights_only=True)
            prefix = "base_model.model." + record["module"].removeprefix("model.")
            tensors = factors[request["variant"]]
            a, b = tensors[prefix + ".lora_A.weight"].double(), tensors[prefix + ".lora_B.weight"].double()
            config = json.loads((weights / f"adapter-{request['variant']}/adapter_config.json").read_text())
            b = b * (config["lora_alpha"] / config["r"])
            shrink = record["input"].double() @ a.T
            ideal = (shrink @ b.T).to(record["raw_delta"].dtype)
            rounded = (shrink.bfloat16().double() @ b.T).to(record["raw_delta"].dtype)
            results.append(
                {
                    "label": request["label"],
                    "module": record["module"],
                    "step": record["step"],
                    "fp64": compare(record["raw_delta"], ideal, contract["atol"], contract["rtol"]),
                    "bf16_intermediate": compare(record["raw_delta"], rounded, contract["atol"], contract["rtol"]),
                }
            )
    report = {
        "purpose": "diagnostic only; the frozen acceptance criteria remain unchanged",
        "records": results,
        "fp64_failed": sum(not r["fp64"]["passed"] for r in results),
        "bf16_intermediate_failed": sum(not r["bf16_intermediate"]["passed"] for r in results),
    }
    (output / "o-intermediate-control.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.weights, args.output)
