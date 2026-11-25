# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Reduced DeepSeek-V4.1 IPC coverage using the shared TP2 transaction."""

from dataclasses import replace

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import (
    DEEPSEEK_V41_CASES,
    WeightUpdateModelCase,
    pytest_model_cases,
)
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import (
    run_npu_ipc_weight_transfer_transaction,
)


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=DEEPSEEK_V41_CASES))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool, deepseek_v41_checkpoint):
    run_npu_ipc_weight_transfer_transaction(replace(case, model=deepseek_v41_checkpoint), packed)
