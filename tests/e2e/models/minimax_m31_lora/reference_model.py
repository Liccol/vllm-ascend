# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Independent installed Transformers M3 FP32 teacher forcing, without merged weights."""

import argparse
import inspect
import json
from pathlib import Path

import torch
from reference_checks import compare
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText


def run(weights, output):
    torch.set_num_threads(8)
    contract = json.loads((output / "contract.json").read_text())
    requests = json.loads((output / "requests.json").read_text())
    model, loading_info = AutoModelForImageTextToText.from_pretrained(
        weights / "base",
        dtype=torch.float32,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        output_loading_info=True,
    )
    model.eval()
    missing = {key: value for key, value in loading_info.items() if value}
    identity = {"class": type(model).__name__, "source": inspect.getsourcefile(type(model)), "loading_issues": missing}
    (output / "hf-identity.json").write_text(json.dumps(identity, indent=2))
    if missing:
        raise ValueError("Independent HF model has missing, unexpected, or mismatched checkpoint tensors")
    index = json.loads((weights / "base/model.safetensors.index.json").read_text())["weight_map"]
    targets = {}
    for name, module in model.named_modules():
        if "vision" not in name and name.endswith(tuple(f".self_attn.{s}_proj" for s in ("q", "k", "v", "o"))):
            canonical = "language_model.model.layers." + name.split("layers.", 1)[1]
            with safe_open(weights / "base" / index[canonical + ".weight"], framework="pt") as handle:
                expected = handle.get_tensor(canonical + ".weight").float()
            if not torch.equal(module.weight, expected):
                raise ValueError(f"Independent HF target does not match source checkpoint: {canonical}")
            targets[canonical] = module
    if len(targets) != 4 * model.config.text_config.num_hidden_layers:
        raise ValueError("Independent HF model did not expose every declared attention target")
    results = []
    for row in requests:
        if row["variant"] not in {"base", "zero", "a", "b"} or not row["label"].endswith(f"p{row['prompt_id']}"):
            continue
        handles = []
        if row["variant"] != "base":
            path = Path(contract["adapters"][row["variant"]]["path"])
            tensors = load_file(path / "adapter_model.safetensors")
            adapter_config = json.loads((path / "adapter_config.json").read_text())
            scale = adapter_config["lora_alpha"] / adapter_config["r"]
            for name, module in targets.items():
                prefix = "base_model.model." + name
                a, b = tensors[prefix + ".lora_A.weight"].float(), tensors[prefix + ".lora_B.weight"].float()

                def add_delta(current, inputs, value, a=a, b=b, scale=scale):
                    return value + scale * (inputs[0].float() @ a.T) @ b.T

                handles.append(module.register_forward_hook(add_delta))
        try:
            generated = row["token_ids"]
            ids = torch.tensor([row["prompt_token_ids"] + generated[:-1]], dtype=torch.long)
            with torch.inference_mode():
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(generated)).logits[0].float()
            probabilities = logits.log_softmax(-1)
            actual, expected = [], []
            for step, sampled in enumerate(row["logprobs"]):
                for token, probability in sampled.items():
                    actual.append(probability)
                    expected.append(float(probabilities[step, int(token)]))
            checked = compare(torch.tensor(actual), torch.tensor(expected), contract["atol"], contract["rtol"])
            results.append(
                {
                    "label": row["label"],
                    "greedy_matches": logits.argmax(-1).tolist() == generated,
                    "logprob": checked,
                    "actual_logprobs": actual,
                    "reference_logprobs": expected,
                    "reference_greedy_tokens": logits.argmax(-1).tolist(),
                }
            )
            (output / "model-reference.json").write_text(json.dumps({"status": "running", "cases": results}, indent=2))
            print("MODEL_REFERENCE", row["label"], checked["passed"], results[-1]["greedy_matches"], flush=True)
        finally:
            for handle in handles:
                handle.remove()
    complete = {row["label"] for row in results} == {
        f"{variant}-p{i}" for variant in ("base", "zero", "a", "b") for i in range(4)
    }
    status = "passed" if all(r["logprob"]["passed"] and r["greedy_matches"] for r in results) else "failed"
    report = {"status": status if complete else "partial", "coverage_complete": complete, "cases": results}
    (output / "model-reference.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.weights, args.output)
    raise SystemExit({"passed": 0, "failed": 1, "partial": 2}[report["status"]])
