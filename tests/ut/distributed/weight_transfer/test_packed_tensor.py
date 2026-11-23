#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""CPU tests for Ascend packed transfer adapters using real tensor operations."""

from __future__ import annotations

import sys
import threading
import types
import weakref
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.weight_transfer.packed_tensor import PackedIpcChunk

from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    NPUPackedBufferImporter,
    packed_broadcast_consumer,
    packed_broadcast_producer,
    packed_npu_ipc_consumer,
    packed_npu_ipc_producer,
)

_MODULE = "vllm_ascend.distributed.weight_transfer.packed_tensor"


class _FakeNpuStream:
    """NPU stream boundary for CPU tests."""

    def __init__(self, *args, **kwargs):
        self.sync_calls = 0

    def wait_stream(self, stream):
        pass

    def wait_event(self, event):
        pass

    def synchronize(self) -> None:
        self.sync_calls += 1
        return None

    def __enter__(self) -> _FakeNpuStream:
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture(autouse=True)
def _stub_torch_npu():
    """Stub device primitives and isolate the global APIs and upstream cache."""
    caller = _FakeNpuStream()
    active = caller

    @contextmanager
    def stream_context(stream):
        nonlocal active
        previous, active = active, stream
        try:
            yield
        finally:
            active = previous

    fake_npu = MagicMock(
        Stream=_FakeNpuStream,
        Event=MagicMock,
        current_device=lambda: 0,
        current_stream=lambda: active,
        synchronize=lambda *a, **kw: None,
        # Match torch.npu.stream by restoring the previous stream on exit.
        stream=stream_context,
    )
    original_empty = torch.empty

    def _fake_empty(*args, **kwargs):
        device = kwargs.get("device")
        if getattr(device, "type", None) == "npu" or (isinstance(device, str) and device.split(":", 1)[0] == "npu"):
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    from vllm.distributed.weight_transfer import packed_tensor as upstream

    from vllm_ascend.worker.v2 import utils as v2_utils

    upstream._get_streams.cache_clear()
    try:
        with (
            patch.dict(torch.cuda.__dict__),
            patch.dict(v2_utils.breakable_cudagraph.__dict__),
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(torch, "empty", _fake_empty),
            patch.object(torch.accelerator, "current_device_index", return_value=0),
            patch.object(torch.accelerator, "current_stream", side_effect=lambda *args, **kwargs: active),
            patch.object(torch, "Event", MagicMock()),
            patch.object(torch.Tensor, "record_stream", return_value=None),
        ):
            yield
    finally:
        # Teardown must clear too: cached fake streams would otherwise survive
        # the patched-API restore and leak into subsequent tests.
        upstream._get_streams.cache_clear()


def test_packed_ipc_rejects_oversized_tensor():
    with (
        patch(f"{_MODULE}.reduce_tensor", return_value=(None, (0,) * 7)),
        pytest.raises(ValueError, match="exceeds buffer_size_bytes"),
    ):
        list(packed_npu_ipc_producer(iter([("big", torch.zeros(4))]), "node-0", lambda item: item[1], 8))


def _install_fake_rebuild_npu_tensor(rebuilt: torch.Tensor | None = None):
    """Install a fake ``torch_npu.multiprocessing.reductions`` module."""
    fake_mod = types.ModuleType("torch_npu.multiprocessing.reductions")
    fake_mod.rebuild_npu_tensor = MagicMock(return_value=rebuilt)  # type: ignore[attr-defined]
    return patch.dict(
        "sys.modules",
        {
            "torch_npu.multiprocessing": types.ModuleType("torch_npu.multiprocessing"),
            "torch_npu.multiprocessing.reductions": fake_mod,
        },
    )


@pytest.mark.parametrize("backend", ["hccl", "ipc"])
@pytest.mark.parametrize(
    "case,buffer_size",
    [("single-chunk", 64), ("multi-chunk", 16), ("empty-iterator", 16), ("zero-elements", 16)],
)
def test_packed_roundtrip(backend, case, buffer_size):
    """Exercise callbacks, byte layout and ownership through the public APIs."""
    # Three BF16 elements put INT64 at the unaligned byte offset 6.
    tensors = [
        ("bf16", torch.tensor([1, 2, 3], dtype=torch.bfloat16)),
        ("int64", torch.tensor([7], dtype=torch.int64)),
        ("matrix", torch.arange(6, dtype=torch.float16).reshape(2, 3).t()),
        ("scalar", torch.tensor(2.5)),
        ("empty", torch.empty(0, 3)),
        ("byte", torch.tensor([7], dtype=torch.uint8)),
    ]
    if case == "empty-iterator":
        tensors = []
    elif case == "zero-elements":
        tensors = [("empty", torch.empty(0, 3))]
    buffers, received = [], []

    def prepare(item):
        return item[1] * 2

    if backend == "hccl":
        group = MagicMock()
        packed_broadcast_producer(iter(tensors), group, 3, prepare, buffer_size)
        buffers.extend(call.args[0] for call in group.broadcast.call_args_list)
        chunk_count = len(buffers)
        assert [call.kwargs["src"] for call in group.broadcast.call_args_list] == [3] * chunk_count
        payloads = iter(buffers.copy())

        def receive(packed, src, stream):
            assert src == 3
            packed.copy_(next(payloads))
            buffers.append(packed)

        packed_broadcast_consumer(
            iter((name, (list(tensor.shape), tensor.dtype)) for name, tensor in tensors),
            types.SimpleNamespace(broadcast=receive),
            3,
            received.extend,
            buffer_size,
        )
    else:
        args = (0,) * 6 + (7,)

        def export(packed):
            buffers.append(packed)
            return None, args

        importer = NPUPackedBufferImporter()
        chunk_count = 0
        with patch(f"{_MODULE}.reduce_tensor", side_effect=export), _install_fake_rebuild_npu_tensor():
            rebuild = sys.modules["torch_npu.multiprocessing.reductions"].rebuild_npu_tensor
            rebuild.side_effect = lambda *args: buffers[0]
            # Read each yield before advancing: the producer reuses its buffer.
            for chunk in packed_npu_ipc_producer(iter(tensors), "node-0", prepare, buffer_size):
                assert type(chunk) is PackedIpcChunk
                assert chunk.ipc_handle == {"node-0": args}
                assert sum(chunk.tensor_sizes) <= buffer_size
                chunk_count += 1
                received.extend(
                    packed_npu_ipc_consumer(
                        chunk.ipc_handle,
                        "node-0",
                        chunk.names,
                        chunk.shapes,
                        chunk.dtype_names,
                        chunk.tensor_sizes,
                        5,
                        importer,
                    )
                )
            assert len(buffers) == 1
            assert rebuild.call_count == bool(chunk_count)
            if chunk_count:
                assert rebuild.call_args.args[6] == 5
        importer.close()

    if case == "empty-iterator":
        assert chunk_count == 0
    elif case == "multi-chunk":
        assert chunk_count > 1
    else:
        assert chunk_count == 1
    for buffer in buffers:
        buffer.zero_()
    assert [name for name, _ in received] == [name for name, _ in tensors]
    for (_, actual), (_, expected) in zip(received, tensors):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected * 2)
    pointers = [tensor.untyped_storage().data_ptr() for _, tensor in received if tensor.numel()]
    assert len(set(pointers)) == len(pointers)
    assert set(pointers).isdisjoint(buffer.untyped_storage().data_ptr() for buffer in buffers if buffer.numel())


@pytest.mark.parametrize("producer", [False, True])
def test_packed_transfer_drains_storage_before_propagating_error(producer):
    import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_module

    events: list[str] = []
    pinned: list[torch.Tensor] = []

    class TrackingStream(_FakeNpuStream):
        buffer = None
        pending = None

        def synchronize(self):
            if self.pending is not None:
                events.append("drain")
                self.pending = None

    stream = TrackingStream()

    def broadcast(packed, **kwargs):
        stream.buffer = weakref.ref(packed)
        if producer:
            stream.pending = stream.buffer
            events.append("error")
            raise ValueError("transfer rejected")

    def reject_load(weights):
        # Receive-before-unpack synchronization must not satisfy this check:
        # only finally can drain work queued by the failing loader.
        stream.pending = stream.buffer
        events.append("error")
        raise ValueError("transfer rejected")

    group = types.SimpleNamespace(broadcast=broadcast)
    with (
        patch.object(torch.npu, "Stream", return_value=stream),
        patch.object(packed_module, "_UNSYNCHRONIZED_BUFFERS", pinned),
        pytest.raises(ValueError, match="transfer rejected"),
    ):
        if producer:
            packed_broadcast_producer(iter([("w", torch.ones(1))]), group, 0, lambda item: item[1], 4, 1)
        else:
            packed_broadcast_consumer(
                iter([("w", ([1], torch.float32))]),
                group,
                0,
                reject_load,
                4,
                1,
            )
    assert events == ["error", "drain"]
    assert pinned == []


def test_packed_consumer_waits_for_receive_before_unpacking():
    """An external HCCL receive must finish before torch queues unpack operations."""
    events = []
    original = torch.tensor([2.5])

    class ReceiveStream(_FakeNpuStream):
        pending = None

        def synchronize(self):
            if self.pending is not None:
                self.pending()
                self.pending = None
                events.append("ready")

    stream = ReceiveStream()

    def receive(packed, **kwargs):
        assert kwargs["stream"] is stream
        stream.pending = lambda: packed.copy_(original.view(torch.uint8))
        events.append("receive")

    def load(weights):
        assert events == ["receive", "ready"]
        assert torch.equal(weights[0][1], original)
        events.append("load")

    with patch.object(torch.npu, "Stream", return_value=stream):
        packed_broadcast_consumer(
            iter([("w", ([1], torch.float32))]), types.SimpleNamespace(broadcast=receive), 0, load, 4, 1
        )
    assert events == ["receive", "ready", "load"]


@pytest.mark.parametrize("backend", ["hccl", "ipc"])
@pytest.mark.parametrize("change", ["dtype", "numel", "none"])
def test_packed_producer_rejects_invalid_callback_result(backend, change):
    original = torch.ones(2)
    returned = {"dtype": original.to(torch.int32), "numel": original[:1], "none": None}[change]
    message = " returned.*expected dtype=.*numel=2" if change in ("dtype", "numel") else ".*must return a torch.Tensor"
    group = MagicMock()
    with (
        patch(f"{_MODULE}.reduce_tensor", return_value=(None, (0,) * 7)),
        pytest.raises(ValueError, match=f"post_iter_func{message}"),
    ):
        if backend == "hccl":
            packed_broadcast_producer(iter([("w", original)]), group, 0, lambda _: returned, 16)
        else:
            list(packed_npu_ipc_producer(iter([("w", original)]), "node-0", lambda _: returned, 16))
    group.broadcast.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"device_index": -1}, id="negative-device"),
        pytest.param({"device_index": None}, id="noninteger-device"),
        pytest.param({"device_index": True}, id="bool-device"),
        pytest.param({"ipc_handle": {"other": (0,) * 7}}, id="missing-uuid"),
        pytest.param({"ipc_handle": {"node-0": (0,)}}, id="short-handle"),
        pytest.param({"dtype_names": ["not_a_dtype"]}, id="unknown-dtype"),
        pytest.param({"dtype_names": ["Tensor"]}, id="non-dtype"),
        pytest.param({"names": ["w", "extra"]}, id="lengths"),
        pytest.param({"shapes": [[-1]]}, id="negative-shape"),
        pytest.param({"tensor_sizes": [-4]}, id="negative-size"),
        pytest.param({"tensor_sizes": [8]}, id="byte-count"),
    ],
)
def test_packed_ipc_consumer_rejects_invalid_metadata_before_rebuild(changes):
    kwargs: dict[str, Any] = dict(
        ipc_handle={"node-0": (0,) * 7},
        physical_npu_id="node-0",
        names=["w"],
        shapes=[[1]],
        dtype_names=["float32"],
        tensor_sizes=[4],
        device_index=0,
    )
    kwargs.update(changes)
    with _install_fake_rebuild_npu_tensor(torch.zeros(8, dtype=torch.uint8)):
        rebuild = sys.modules["torch_npu.multiprocessing.reductions"].rebuild_npu_tensor
        with pytest.raises(ValueError):
            packed_npu_ipc_consumer(**kwargs)
        rebuild.assert_not_called()


def test_packed_hccl_consumer_rejects_negative_shape_before_broadcast():
    group = MagicMock()
    with pytest.raises(ValueError, match="negative shape"):
        packed_broadcast_consumer(iter([("w", ([-1], torch.float32))]), group, 0, MagicMock(), 4)
    group.broadcast.assert_not_called()


@pytest.mark.parametrize(
    "producer,logging_error",
    [
        (False, None),
        (True, RuntimeError("logging failed")),
        (False, KeyboardInterrupt()),
        (True, SystemExit(2)),
    ],
)
def test_packed_drain_failure_keeps_storage_and_original_error(producer, logging_error):
    import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_module

    class FailedStream(_FakeNpuStream):
        failed = False

        def synchronize(self):
            if self.failed:
                raise RuntimeError("drain failed")

    stream = FailedStream()
    pinned: list[torch.Tensor] = []
    original_error = ValueError("transfer rejected")

    def broadcast(*args, **kwargs):
        stream.failed = True
        raise original_error

    expected_type = ValueError if logging_error is None or isinstance(logging_error, Exception) else type(logging_error)
    with (
        patch.object(torch.npu, "Stream", return_value=stream),
        patch.object(packed_module, "_UNSYNCHRONIZED_BUFFERS", pinned),
        patch.object(packed_module.logger, "error", side_effect=logging_error) as log_error,
        pytest.raises(expected_type) as error,
    ):
        group = types.SimpleNamespace(broadcast=broadcast)
        if producer:
            packed_broadcast_producer(iter([("w", torch.ones(1))]), group, 0, lambda item: item[1], 4, 1)
        else:
            packed_broadcast_consumer(iter([("w", ([1], torch.float32))]), group, 0, MagicMock(), 4, 1)
    if logging_error is None or isinstance(logging_error, Exception):
        assert error.value is original_error
    else:
        assert error.value is logging_error
    assert len(pinned) == 1
    assert pinned[0].numel() == 4
    log_error.assert_called_once()
    message, retained_slots, cleanup_error = log_error.call_args.args
    assert "(slot, bytes)" in message
    assert retained_slots == [(0, 4)]
    assert str(cleanup_error) == "drain failed"
    assert not log_error.call_args.kwargs


def test_packed_ipc_consumer_rejects_payload_larger_than_rebuilt_buffer():
    with (
        _install_fake_rebuild_npu_tensor(torch.zeros(3, dtype=torch.uint8)),
        pytest.raises(ValueError, match="metadata declares 4 bytes, buffer has 3"),
    ):
        packed_npu_ipc_consumer({"node-0": (0,) * 7}, "node-0", ["w"], [[1]], ["float32"], [4], 0)


@pytest.mark.parametrize(
    "backend,options",
    [
        ("hccl-producer", {"buffer_size_bytes": 0}),
        ("hccl-consumer", {"buffer_size_bytes": 0}),
        ("ipc-producer", {"buffer_size_bytes": -1}),
        ("hccl-producer", {"num_buffers": 0}),
        ("hccl-consumer", {"num_buffers": -1}),
    ],
)
def test_packed_rejects_invalid_buffer_configuration(backend, options):
    with pytest.raises(ValueError, match="positive"):
        if backend == "hccl-producer":
            packed_broadcast_producer(iter([]), MagicMock(), 0, lambda item: item[1], **options)
        elif backend == "hccl-consumer":
            packed_broadcast_consumer(iter([]), MagicMock(), 0, MagicMock(), **options)
        else:
            list(packed_npu_ipc_producer(iter([]), "node-0", lambda item: item[1], **options))


@pytest.mark.parametrize("num_buffers", [1, 2, 3])
def test_packed_producer_passes_packing_stream_to_broadcast(num_buffers):
    pending: list[_FakeNpuStream] = []
    packed_items = 0

    class BroadcastStream(_FakeNpuStream):
        def synchronize(self):
            if pending == [self]:
                pending.clear()

    streams = [BroadcastStream() for _ in range(num_buffers)]
    active: list[_FakeNpuStream] = []

    class StreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            active.append(self.stream)

        def __exit__(self, *exc):
            active.pop()

    def broadcast(tensor, *, src, stream):
        assert active == [stream]
        assert not pending, "previous broadcast must complete before the next collective"
        pending.append(stream)

    def prepare(item):
        nonlocal packed_items
        # Another slot can pack while the previous transfer is still pending.
        # A single slot must finish its own transfer before it can be reused.
        assert bool(pending) == (packed_items > 0 and num_buffers > 1)
        packed_items += 1
        return item[1]

    group = types.SimpleNamespace(broadcast=MagicMock(side_effect=broadcast))
    with (
        patch.object(torch.npu, "Stream", side_effect=streams),
        patch.object(torch.npu, "stream", side_effect=StreamContext) as context,
        patch.object(torch.npu, "current_stream", side_effect=lambda: active[-1] if active else streams[0]),
    ):
        packed_broadcast_producer(iter((f"w{i}", torch.ones(1)) for i in range(7)), group, 0, prepare, 1, num_buffers)
    expected = [streams[i % num_buffers] for i in range(7)]
    assert [call.kwargs["stream"] for call in group.broadcast.call_args_list] == expected
    assert [call.args[0] for call in context.call_args_list[:7]] == expected
    assert active == []
    assert pending == []
    assert packed_items == 7


@pytest.mark.parametrize("buffer_size", [4, 32], ids=["full-chunk", "final-chunk"])
def test_packed_consumer_does_not_retry_receive_when_loader_raises_stop_iteration(buffer_size):
    failure = StopIteration("loader failed")
    loaded = []

    def load(weights):
        loaded.append([name for name, _ in weights])
        raise failure

    group = MagicMock()
    with pytest.raises(RuntimeError, match="post_unpack_func raised StopIteration") as error:
        packed_broadcast_consumer(
            iter([("a", ([1], torch.float32)), ("b", ([1], torch.float32))]),
            group,
            0,
            load,
            buffer_size,
            2,
        )
    assert error.value.__cause__ is failure
    assert loaded == [["a", "b"]]
    assert group.broadcast.call_count == 1


@pytest.mark.parametrize(
    "producer,body_error,second_error",
    [
        (False, None, None),
        (True, None, KeyboardInterrupt()),
        (False, None, SystemExit(2)),
        (True, ValueError("body failed"), KeyboardInterrupt()),
        (False, ValueError("body failed"), SystemExit(2)),
        (True, KeyboardInterrupt(), None),
        (False, SystemExit(2), None),
    ],
)
def test_packed_cleanup_drains_all_slots_reports_retention_and_prioritizes_signals(producer, body_error, second_error):
    import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_module

    failure = RuntimeError("cleanup failed")
    failures = [failure, second_error, None]
    drained = []
    armed = False

    class CleanupStream(_FakeNpuStream):
        def __init__(self, index):
            self.index = index

        def synchronize(self):
            if armed:
                drained.append(self.index)
                if failures[self.index] is not None:
                    raise failures[self.index]

    streams = [CleanupStream(index) for index in range(3)]
    pinned: list[torch.Tensor] = []

    def finish(*args, **kwargs):
        nonlocal armed
        armed = True
        if body_error is not None:
            raise body_error

    errors = [error for error in (body_error, *failures) if error is not None]
    expected = next((error for error in errors if not isinstance(error, Exception)), errors[0])
    group = types.SimpleNamespace(broadcast=finish if producer else lambda *args, **kwargs: None)
    with (
        patch.object(torch.npu, "Stream", side_effect=streams),
        patch.object(packed_module, "_UNSYNCHRONIZED_BUFFERS", pinned),
        patch.object(packed_module.logger, "error") as log_error,
        pytest.raises(type(expected)) as error,
    ):
        # One tensor below the threshold reaches finally without another slot wait.
        if producer:
            packed_broadcast_producer(iter([("w", torch.ones(1))]), group, 0, lambda item: item[1], 8, 3)
        else:
            packed_broadcast_consumer(iter([("w", ([1], torch.float32))]), group, 0, finish, 8, 3)
    assert error.value is expected
    assert drained[-3:] == [0, 1, 2]
    assert len(pinned) == 1
    assert pinned[0].numel() == 4
    log_error.assert_called_once()
    assert log_error.call_args.args[1] == [(0, 4)]


@pytest.mark.parametrize("producer", [False, True])
def test_packed_drain_failure_does_not_retain_unused_empty_slots(producer):
    import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_module

    class FailedStream(_FakeNpuStream):
        def synchronize(self):
            raise RuntimeError("stream unavailable")

    pinned: list[torch.Tensor] = []
    with (
        patch.object(torch.npu, "Stream", return_value=FailedStream()),
        patch.object(packed_module, "_UNSYNCHRONIZED_BUFFERS", pinned),
        pytest.raises(RuntimeError, match="stream unavailable"),
    ):
        if producer:
            packed_broadcast_producer(iter([]), MagicMock(), 0, lambda item: item[1], 4, 2)
        else:
            packed_broadcast_consumer(iter([]), MagicMock(), 0, MagicMock(), 4, 2)
    assert pinned == []


@pytest.mark.parametrize("producer", [False, True], ids=["consumer", "producer"])
@pytest.mark.parametrize("num_buffers", [1, 3])
def test_packed_normal_path_drains_every_slot_stream(producer, num_buffers):
    events = []
    original_context = torch.npu.stream

    class Stream(_FakeNpuStream):
        def synchronize(self):
            events.append(("sync", self))

    @contextmanager
    def context(stream):
        with original_context(stream):
            yield
        events.append(("exit", stream))

    streams = [Stream() for _ in range(num_buffers)]
    with patch.object(torch.npu, "Stream", side_effect=streams), patch.object(torch.npu, "stream", context):
        if producer:
            packed_broadcast_producer(
                iter([("w", torch.ones(1))]), MagicMock(), 0, lambda item: item[1], 8, num_buffers
            )
        else:
            packed_broadcast_consumer(iter([("w", ([1], torch.float32))]), MagicMock(), 0, MagicMock(), 8, num_buffers)
    last_exit = max(index for index, (kind, _) in enumerate(events) if kind == "exit")
    assert events[last_exit + 1 :] == [("sync", stream) for stream in streams]


@pytest.mark.parametrize("producer", [False, True])
def test_upstream_stream_cache_drift_fails_before_collective(producer):
    """A changed cache contract must not submit HCCL on an unregistered stream."""
    from vllm.distributed.weight_transfer import packed_tensor as upstream

    group = MagicMock()
    with (
        patch.object(
            upstream, "_get_streams", side_effect=lambda device, count: tuple(_FakeNpuStream() for _ in range(count))
        ),
        pytest.raises(RuntimeError, match="packed stream cache changed"),
    ):
        if producer:
            packed_broadcast_producer(iter([("w", torch.ones(1))]), group, 0, lambda item: item[1], 8, 2)
        else:
            packed_broadcast_consumer(iter([("w", ([1], torch.float32))]), group, 0, lambda weights: None, 8, 2)
    group.broadcast.assert_not_called()


@pytest.mark.parametrize("producer", [False, True], ids=["consumer", "producer"])
def test_concurrent_hccl_transfer_is_rejected_before_collective(producer):
    entered, release = threading.Event(), threading.Event()
    errors = []

    def hold_broadcast(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)

    def active_transfer():
        try:
            packed_broadcast_producer(
                iter([("active", torch.ones(1))]),
                types.SimpleNamespace(broadcast=hold_broadcast),
                0,
                lambda item: item[1],
                8,
                1,
            )
        except BaseException as exc:
            errors.append(exc)

    group = MagicMock()

    def second_transfer():
        # A different slot count must not bypass the device-level guard.
        if producer:
            packed_broadcast_producer(iter([("w", torch.ones(1))]), group, 0, lambda item: item[1], 8, 2)
        else:
            group.broadcast.side_effect = lambda tensor, **kwargs: tensor.fill_(0)
            packed_broadcast_consumer(iter([("w", ([1], torch.float32))]), group, 0, lambda weights: None, 8, 2)

    thread = threading.Thread(target=active_transfer, daemon=True)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(RuntimeError, match="must be serialized across communicators"):
            second_transfer()
        group.broadcast.assert_not_called()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    second_transfer()
    group.broadcast.assert_called_once()


def test_hccl_transfer_guard_is_scoped_to_device_and_released_on_error():
    import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_module

    with packed_module._hccl_group(MagicMock(), 1):
        with patch.object(torch.accelerator, "current_device_index", return_value=1):
            with (
                pytest.raises(RuntimeError, match="loader failed"),
                packed_module._hccl_group(MagicMock(), 2),
            ):
                raise RuntimeError("loader failed")
            with packed_module._hccl_group(MagicMock(), 1):
                pass
        with (
            pytest.raises(RuntimeError, match="must be serialized"),
            packed_module._hccl_group(MagicMock(), 1),
        ):
            pass


@pytest.mark.parametrize("change", ["missing", "signature", "count"])
def test_incompatible_upstream_stream_api_fails_before_collective(monkeypatch, change):
    from vllm.distributed.weight_transfer import packed_tensor as upstream

    if change == "missing":
        monkeypatch.delattr(upstream, "_get_streams")
    elif change == "signature":
        monkeypatch.setattr(upstream, "_get_streams", lambda device: ())
    else:
        monkeypatch.setattr(upstream, "_get_streams", lambda device, count: ())
    group = MagicMock()
    with pytest.raises(RuntimeError, match="Incompatible vLLM packed stream"):
        packed_broadcast_producer(iter([("w", torch.ones(1))]), group, 0, lambda item: item[1], 8, 2)
    group.broadcast.assert_not_called()
