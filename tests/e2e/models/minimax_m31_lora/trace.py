# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in worker lifecycle extension for batch evidence, not performance runs."""

import json
from collections import Counter
from pathlib import Path


class LoRABatchTrace:
    def init_worker_extension(self):
        from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

        original = PunicaWrapperNPU.update_metadata
        rank = self.rank
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        marker = Path.cwd() / ".minimax-lora-trace.json"

        def update(wrapper, mapping, slot_ids, *args, **kwargs):
            # The client enables recording after readiness, excluding warmup.
            if marker.exists():
                directory = Path(json.loads(marker.read_text())["directory"])
                row = {
                    "dp_rank": dp_rank,
                    "rank": rank,
                    "tokens": dict(Counter(int(value) for value in mapping.index_mapping)),
                    "slots": list(slot_ids),
                    "prompts": [int(value) for value in mapping.prompt_mapping],
                }
                with (directory / f"dp{dp_rank}-rank{rank}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
            return original(wrapper, mapping, slot_ids, *args, **kwargs)

        PunicaWrapperNPU.update_metadata = update
