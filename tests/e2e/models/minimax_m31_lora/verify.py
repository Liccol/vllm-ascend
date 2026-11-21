# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Capture native token IDs and fail on merged/serial/mixed serving mismatches."""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "Please compute 17 + 25. Answer:",
    "def add(a, b):\n    return",
    "Summarize this passage: " + "A library stores books and lends them to readers. " * 24,
]


def capture(url, tokenizer_path, models, max_tokens, rounds):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompts = [tokenizer.encode(p, add_special_tokens=False) for p in PROMPTS]
    jobs = [(model, i) for model in models for i in range(len(prompts))]

    def request(job, barrier=None):
        model, prompt_id = job
        if barrier is not None:
            barrier.wait(timeout=30)
        response = requests.post(
            url.rstrip("/") + "/v1/completions",
            json={
                "model": model,
                "prompt": prompts[prompt_id],
                "temperature": 0,
                "top_p": 1,
                "seed": 0,
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "n": 1,
                "return_token_ids": True,
                "logprobs": 5,
                "stream": False,
            },
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()
        choice = result["choices"][0]
        if not choice.get("token_ids"):
            raise ValueError("Server did not return native token_ids")
        if choice.get("prompt_token_ids") != prompts[prompt_id]:
            raise ValueError("Server input token IDs differ from the fixture")
        return {
            "model": model,
            "prompt_id": prompt_id,
            "request_id": result["id"],
            "prompt_token_ids": prompts[prompt_id],
            "token_ids": choice["token_ids"],
            "finish_reason": choice["finish_reason"],
            "logprobs": choice.get("logprobs"),
        }

    serial = [request(job) for job in jobs]
    mixed = []
    for number in range(rounds):
        ordered = jobs[number % len(jobs) :] + jobs[: number % len(jobs)]
        if number % 2:
            ordered = list(reversed(ordered))
        barrier = threading.Barrier(len(ordered))
        with ThreadPoolExecutor(max_workers=len(ordered)) as pool:
            mixed.append(list(pool.map(lambda job, barrier=barrier: request(job, barrier), ordered)))
    return {"max_tokens": max_tokens, "serial": serial, "mixed": mixed}


def compare(reference, actual):
    failures = []
    by_prompt = {row["prompt_id"]: row for row in reference}
    for row in actual:
        expected = by_prompt[row["prompt_id"]]
        if row["prompt_token_ids"] != expected["prompt_token_ids"]:
            raise ValueError("Cannot compare different input token IDs")
        left, right = expected["token_ids"], row["token_ids"]
        if left != right or expected["finish_reason"] != row["finish_reason"]:
            first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            failures.append(
                {
                    "model": row["model"],
                    "prompt_id": row["prompt_id"],
                    "first_difference": first,
                    "expected_token_ids": left,
                    "actual_token_ids": right,
                    "expected_finish_reason": expected["finish_reason"],
                    "actual_finish_reason": row["finish_reason"],
                }
            )
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--models", nargs="+", default=["base", "lora-a", "lora-b"])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trace-dir", type=Path, help="Client must share the worker filesystem and working directory")
    parser.add_argument("--expected-dp", type=int, default=2)
    parser.add_argument("--reference-base", type=Path)
    parser.add_argument("--reference-a", type=Path)
    parser.add_argument("--reference-b", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=False)
        marker = Path.cwd() / ".minimax-lora-trace.json"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"directory": str(args.trace_dir.resolve())}))
        temporary.replace(marker)
    try:
        result = capture(args.url, args.tokenizer, args.models, args.max_tokens, args.rounds)
    finally:
        if args.trace_dir:
            marker.unlink(missing_ok=True)
    failures = []
    for model in args.models:
        serial = [row for row in result["serial"] if row["model"] == model]
        for batch in result["mixed"]:
            failures.extend(compare(serial, [row for row in batch if row["model"] == model]))
    for model, path in (("base", args.reference_base), ("lora-a", args.reference_a), ("lora-b", args.reference_b)):
        if path is not None:
            reference = json.loads(path.read_text())["serial"]
            rows = [row for row in result["serial"] if row["model"] == model]
            if not rows:
                raise ValueError(f"No actual rows for reference model {model}")
            failures.extend(compare(reference, rows))
    if {"base", "lora-a", "lora-b"}.issubset(args.models):
        serial = {
            model: {row["prompt_id"]: row for row in result["serial"] if row["model"] == model}
            for model in ("base", "lora-a", "lora-b")
        }
        result["discriminating_prompts"] = {}
        for left, right in (("base", "lora-a"), ("base", "lora-b"), ("lora-a", "lora-b")):
            different = [i for i in serial[left] if serial[left][i]["token_ids"] != serial[right][i]["token_ids"]]
            result["discriminating_prompts"][left + "/" + right] = different
            if not different:
                failures.append({"reason": f"Fixture cannot distinguish {left} from {right} by generated token IDs"})
    result["failures"] = failures
    if args.trace_dir:
        summary = []
        for path in args.trace_dir.glob("*.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            summary.append(
                {
                    "dp_rank": rows[0]["dp_rank"],
                    "rank": rows[0]["rank"],
                    "updates": len(rows),
                    "base_and_two_adapters": sum({"0", "1", "2"}.issubset(row["tokens"]) for row in rows),
                }
            )
        result["batch_trace"] = summary
        if {row["dp_rank"] for row in summary} != set(range(args.expected_dp)) or not all(
            row["base_and_two_adapters"] > 0 for row in summary
        ):
            failures.append({"reason": "Missing real base/A/B mixed-batch evidence on some workers"})
    result["status"] = "pass" if not failures else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": len(failures), "serial_requests": len(result["serial"])}))
    raise SystemExit(bool(failures))
