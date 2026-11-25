# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Local reduced checkpoint required by V4.1's Engram shard reader."""

import pytest
import torch

from tests.e2e.common.rlhf.weight_transfer.case import DEEPSEEK_V41_ENGRAM_CASE
from tests.e2e.common.rlhf.weight_transfer.weight_source import (
    fixed_startup_checkpoint,
    make_weight_source,
)


@pytest.fixture(scope="module")
def deepseek_v41_checkpoint():
    source = make_weight_source(DEEPSEEK_V41_ENGRAM_CASE, torch.device("cpu"))
    with fixed_startup_checkpoint(source) as directory:
        yield directory
