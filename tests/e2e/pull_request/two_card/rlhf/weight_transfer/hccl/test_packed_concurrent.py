# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed HCCL rejects overlapping transfers before entering another communicator."""

import multiprocessing as mp
import os
import queue
import socket
import threading
import time
import traceback

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.distributed.weight_transfer.hccl_common import stateless_init_process_group
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    packed_broadcast_consumer,
    packed_broadcast_producer,
)

_TRANSFER_TIMEOUT = 30
_PROCESS_TIMEOUT = 120


def _concurrent_rank(rank, ports, num_buffers, results):
    try:
        torch.npu.set_device(rank)
        groups = [stateless_init_process_group("127.0.0.1", port, rank, 2, rank) for port in ports]
        for group in groups:
            group.broadcast(torch.ones(4, device="npu"), src=0, stream=torch.npu.current_stream())
            torch.npu.current_stream().synchronize()
        inputs = [[(f"w{j}", torch.full((4,), float(10 * i + j), device="npu")) for j in range(3)] for i in range(2)]
        torch.npu.current_stream().synchronize()
        entered, release = threading.Event(), threading.Event()
        calls = [0, 0]
        received = [[], []]
        errors = []

        class Group:
            def __init__(self, index):
                self.index = index

            def broadcast(self, tensor, src, stream):
                calls[self.index] += 1
                groups[self.index].broadcast(tensor, src=src, stream=stream)
                if self.index == 0:
                    entered.set()
                    assert release.wait(timeout=_TRANSFER_TIMEOUT)

        def transfer(index):
            torch.npu.set_device(rank)
            if rank == 0:
                packed_broadcast_producer(iter(inputs[index]), Group(index), 0, lambda item: item[1], 16, num_buffers)
            else:
                metadata = ((name, ([4], torch.float32)) for name, _ in inputs[index])
                packed_broadcast_consumer(metadata, Group(index), 0, received[index].extend, 16, num_buffers)

        def active_transfer():
            try:
                transfer(0)
            except BaseException:
                errors.append(traceback.format_exc())

        thread = threading.Thread(target=active_transfer, daemon=True)
        thread.start()
        try:
            assert entered.wait(_TRANSFER_TIMEOUT), errors
            with pytest.raises(RuntimeError, match="must be serialized across communicators"):
                transfer(1)
            assert calls[1] == 0
        finally:
            release.set()
            thread.join(timeout=_TRANSFER_TIMEOUT)
        assert not thread.is_alive(), "Active packed transfer stalled"
        assert not errors, errors
        # Rejection must neither poison the other group nor leave the guard set.
        transfer(1)
        assert calls == [2, 2]
        if rank == 1:
            for index, weights in enumerate(received):
                assert [name for name, _ in weights] == ["w0", "w1", "w2"]
                for j, (_, tensor) in enumerate(weights):
                    assert torch.equal(tensor.cpu(), torch.full((4,), float(10 * index + j)))
        for group in groups:
            group.close()
        result = (rank, None)
    except BaseException:
        result = (rank, traceback.format_exc())
    results.put(result)
    results.close()
    results.join_thread()
    if result[1] is not None:
        # Avoid driver teardown when a timed-out collective may still be pending.
        os._exit(1)


@pytest.mark.parametrize("num_buffers", [1, 2])
def test_overlapping_communicators_rejected_then_reused(num_buffers):
    ctx = mp.get_context("spawn")
    # Reserve both ports together so they cannot accidentally be identical.
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        ports = [first.getsockname()[1], second.getsockname()[1]]
    results = ctx.Queue()
    processes = [ctx.Process(target=_concurrent_rank, args=(rank, ports, num_buffers, results)) for rank in range(2)]
    reports = []
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + _PROCESS_TIMEOUT
        for _ in processes:
            try:
                reports.append(results.get(timeout=max(0, deadline - time.monotonic())))
            except queue.Empty:
                pytest.fail(f"NPU worker timed out; reports: {reports}")
        assert sorted(rank for rank, _ in reports) == [0, 1]
        assert all(error is None for _, error in reports), reports
    finally:
        for process in processes:
            if process.pid is None:
                continue
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        results.close()
