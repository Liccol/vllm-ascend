# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise retained layer inputs across real NPU slot streams."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.distributed.weight_transfer.packed_tensor import packed_broadcast_consumer

_DELAY_MATRIX_SIZE = 4096
_DELAY_MATMUL_COUNT = 256
_RETAINED_WEIGHT_ELEMENTS = 128 * 1024
_ALLOCATOR_REUSE_ATTEMPTS = 64


@pytest.mark.parametrize("num_buffers", [1, 2])
def test_layer_spanning_chunks_reads_completed_clones(num_buffers):
    torch.npu.set_device(0)
    expected = torch.tensor([1.0, 2.0], device="npu")
    payloads = iter(expected.split(1))
    output = torch.zeros_like(expected)
    torch.npu.synchronize()
    retained = []
    ready = torch.npu.Event()
    clone_count = 0
    original_clone = torch.Tensor.clone
    matrix = torch.ones((_DELAY_MATRIX_SIZE, _DELAY_MATRIX_SIZE), device="npu", dtype=torch.float16)
    scratch = torch.empty_like(matrix)
    torch.mm(matrix, matrix, out=scratch)
    torch.npu.synchronize()

    def delayed_clone(raw, *args, **kwargs):
        nonlocal clone_count
        clone_count += 1
        if clone_count != 1:
            return original_clone(raw, *args, **kwargs)
        result = torch.zeros_like(raw)
        torch.npu.current_stream().synchronize()
        for _ in range(_DELAY_MATMUL_COUNT):
            torch.mm(matrix, matrix, out=scratch)
        result.copy_(raw)
        ready.record(torch.npu.current_stream())
        return result

    def receive(buffer, **kwargs):
        buffer.copy_(next(payloads).view(torch.uint8))

    def load(weights):
        retained.extend(tensor for _, tensor in weights)
        if len(retained) == 2:
            print(f"num_buffers={num_buffers}, clone_pending_at_load={not ready.query()}", flush=True)
            output.copy_(torch.cat(retained))
            torch.npu.current_stream().synchronize()

    with patch.object(torch.Tensor, "clone", delayed_clone):
        packed_broadcast_consumer(
            iter((f"layer.w{i}", ([1], torch.float32)) for i in range(2)),
            SimpleNamespace(broadcast=receive),
            0,
            load,
            1,
            num_buffers,
        )
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)


def test_layer_clone_survives_async_reader_after_loader_releases_it():
    """Separate allocation lifetime from readiness by completing the first clone."""
    torch.npu.set_device(0)
    expected = torch.arange(_RETAINED_WEIGHT_ELEMENTS, dtype=torch.float32, device="npu") + 1
    payloads = iter([expected, torch.ones(1, device="npu")])
    output = torch.zeros_like(expected)
    matrix = torch.ones((_DELAY_MATRIX_SIZE, _DELAY_MATRIX_SIZE), device="npu", dtype=torch.float16)
    scratch = torch.empty_like(matrix)
    torch.mm(matrix, matrix, out=scratch)
    torch.npu.synchronize()
    retained = []
    origin = None
    released_pointer = None
    poison = []

    def receive(buffer, **kwargs):
        buffer.copy_(next(payloads).view(torch.uint8))

    def load(weights):
        nonlocal origin, released_pointer
        if origin is None:
            retained.extend(tensor for _, tensor in weights)
            origin = torch.npu.current_stream()
            return
        # Make input readiness explicit so only allocator lifetime is tested.
        origin.synchronize()
        first = retained.pop()
        released_pointer = first.untyped_storage().data_ptr()
        reader = torch.npu.current_stream()
        for _ in range(_DELAY_MATMUL_COUNT):
            torch.mm(matrix, matrix, out=scratch)
        output.copy_(first)
        del first
        reused = False
        with torch.npu.stream(origin):
            for _ in range(_ALLOCATOR_REUSE_ATTEMPTS):
                replacement = torch.empty_like(expected)
                replacement.zero_()
                poison.append(replacement)
                reused |= replacement.untyped_storage().data_ptr() == released_pointer
            origin.synchronize()
        reader.synchronize()
        print(f"clone_storage_reused={reused}", flush=True)

    packed_broadcast_consumer(
        iter([("layer.w0", ([_RETAINED_WEIGHT_ELEMENTS], torch.float32)), ("layer.w1", ([1], torch.float32))]),
        SimpleNamespace(broadcast=receive),
        0,
        load,
        1,
        2,
    )
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
