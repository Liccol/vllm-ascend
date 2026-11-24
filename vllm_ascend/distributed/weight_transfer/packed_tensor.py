# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import itertools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from threading import Lock
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm.distributed.weight_transfer import packed_tensor as upstream
from vllm.logger import logger

DEFAULT_PACKED_BUFFER_SIZE_BYTES = upstream.DEFAULT_PACKED_BUFFER_SIZE_BYTES
DEFAULT_PACKED_NUM_BUFFERS = upstream.DEFAULT_PACKED_NUM_BUFFERS

# The upstream stream cache is shared by all communicators on a device.
# Track active transfers for their entire lifetime, including failure cleanup.
_HCCL_TRANSFER_LOCK = Lock()
_ACTIVE_HCCL_DEVICES: set[int] = set()


@contextmanager
def _serialized_hccl_transfer(device_idx: int):
    with _HCCL_TRANSFER_LOCK:
        if device_idx in _ACTIVE_HCCL_DEVICES:
            raise RuntimeError(
                f"Packed HCCL transfers on npu:{device_idx} must be serialized across communicators; "
                "wait for the active transfer to finish on every rank before starting another"
            )
        _ACTIVE_HCCL_DEVICES.add(device_idx)
    try:
        yield
    finally:
        with _HCCL_TRANSFER_LOCK:
            _ACTIVE_HCCL_DEVICES.remove(device_idx)


def _get_hccl_streams(device_idx: int, num_buffers: int) -> list[Any]:
    """Keep the private vLLM stream-cache dependency at one compatibility boundary."""
    try:
        streams = list(upstream._get_streams(device_idx, num_buffers))
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("Incompatible vLLM packed stream API; check the supported vLLM pairing") from exc
    if len(streams) != num_buffers:
        raise RuntimeError("Incompatible vLLM packed stream count; check the supported vLLM pairing")
    return streams


# On a failed stream drain, keep storage alive until process exit. Its async
# users may still be running, so returning it to the allocator is unsafe.
# torch_npu's getStreamFromPool uses process-global native streams (see
# NPUStream.cpp / LeakyStreamInternals). Dropping a Python Stream wrapper is
# not a completion barrier. This failed-drain path has no completion proof.
# This failure-only list is intentionally unbounded: retries after a drain
# failure can exhaust device memory. Do not clear it to recover memory while
# work may still be pending. Transfers must not concurrently share a communicator
# or its buffers; this list only holds references, it does not serialize them.
_UNSYNCHRONIZED_BUFFERS: list[torch.Tensor] = []


def _drain_packed_streams(
    streams: list[Any], packed_tensors: list[torch.Tensor], body_error: BaseException | None
) -> None:
    """Drain every slot, retaining failed storage and prioritizing termination."""
    cleanup_error: BaseException | None = None
    retained_slots: list[tuple[int, int]] = []
    for index, stream in enumerate(streams):
        try:
            stream.synchronize()
        except BaseException as exc:
            # Empty placeholders have no payload storage to protect.
            if packed_tensors[index].numel():
                _UNSYNCHRONIZED_BUFFERS.append(packed_tensors[index])
                retained_slots.append((index, packed_tensors[index].numel() * packed_tensors[index].element_size()))
            if cleanup_error is None or (isinstance(cleanup_error, Exception) and not isinstance(exc, Exception)):
                cleanup_error = exc
    if cleanup_error is not None:
        with suppress(Exception):
            logger.error(
                "Failed to drain a packed HCCL stream; "
                "retaining buffers until process exit (slot, bytes): %s; cleanup error: %s",
                retained_slots,
                cleanup_error,
            )
        # A pending body exception propagates unless cleanup raised a higher-
        # priority termination signal. Never skip draining other slots for it.
        if body_error is None or (isinstance(body_error, Exception) and not isinstance(cleanup_error, Exception)):
            raise cleanup_error


class _HCCLPackedGroup:
    def __init__(self, group, streams, *, is_producer=False):
        self.group = group
        self.device = torch.device("npu", torch.accelerator.current_device_index())
        self.streams = streams
        self.buffers = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in streams]
        self.is_producer = is_producer
        self.previous_broadcast_stream = None

    def broadcast(self, tensor, src):
        stream = torch.npu.current_stream()
        try:
            index = self.streams.index(stream)
        except ValueError as exc:
            raise RuntimeError("vLLM packed stream cache changed; check the supported vLLM pairing") from exc
        self.buffers[index] = tensor
        # Flush packing work before native HCCL. Packing on another slot can
        # overlap the previous transfer, but collectives must not overlap.
        stream.synchronize()
        if self.previous_broadcast_stream is not None and self.previous_broadcast_stream is not stream:
            self.previous_broadcast_stream.synchronize()
        self.group.broadcast(tensor, src=src, stream=stream)
        if self.is_producer:
            self.previous_broadcast_stream = stream
        else:
            # Native receive work must finish before torch unpacks the buffer.
            stream.synchronize()


@contextmanager
def _hccl_group(group, num_buffers, *, is_producer=False):
    # Standalone trainers need the same NPU-only API adaptation as V2 workers.
    from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

    device_idx = torch.accelerator.current_device_index()
    with _serialized_hccl_transfer(device_idx), torch_cuda_wrapper():
        # Use the same accelerator index/cache key as the upstream loops.
        streams = _get_hccl_streams(device_idx, num_buffers if is_producer else num_buffers + 1)
        transport = _HCCLPackedGroup(group, streams, is_producer=is_producer)
        try:
            yield transport
        except BaseException as exc:
            _drain_packed_streams(streams, transport.buffers, exc)
            raise


def _checked_callback(callback):
    def checked(item):
        tensor = callback(item)
        original = item[1]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"post_iter_func for {item[0]!r} must return a torch.Tensor")
        if tensor.dtype != original.dtype or tensor.numel() != original.numel():
            raise ValueError(
                f"post_iter_func returned tensor {item[0]!r} with dtype={tensor.dtype}, "
                f"numel={tensor.numel()}; expected dtype={original.dtype}, numel={original.numel()}"
            )
        return tensor

    return checked


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Broadcast tensors in a packed manner from trainer to workers.

    Transfers must be serialized per device on every rank.

    Args:
        iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
        group: Process group (PyHcclCommunicator)
        src: Source rank (0 in current implementation)
        post_iter_func: Function to apply to each (name, tensor) pair before
                       packing. Must return a Tensor with the source dtype and
                       element count, or raises ValueError. Flattened element
                       positions must still match the source metadata: values,
                       device and strides may change, but elements must not be
                       reordered (e.g. by transposing). Order is a caller contract,
                       not checked here.
        buffer_size_bytes: Size in bytes for each packed tensor buffer.
                          Both producer and consumer must use the same value.
        num_buffers: Number of in-flight communication slots.
    """
    if buffer_size_bytes <= 0 or num_buffers <= 0:
        raise ValueError(
            f"buffer_size_bytes and num_buffers must be positive, got "
            f"buffer_size_bytes={buffer_size_bytes}, num_buffers={num_buffers}"
        )
    with _hccl_group(group, num_buffers, is_producer=True) as transport:
        upstream.packed_nccl_broadcast_producer(
            iterator, transport, src, _checked_callback(post_iter_func), buffer_size_bytes, num_buffers
        )


def packed_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Consume packed tensors and unpack them into a list of tensors.

    Output clones keep independent storage. Upstream owns the aligned layout
    and runs all unpacking and callbacks in order on one unpack stream.

    Transfers must be serialized per device on every rank.
    Additional reader streams must arrange their own readiness and storage tracking.

    Args:
        iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
        group: Process group (PyHcclCommunicator)
        src: Source rank (0 in current implementation)
        post_unpack_func: Function to apply to each list of (name, tensor) after
                         unpacking. Tensors own independent storage and may be
                         retained or modified without aliasing the packed buffer.
        buffer_size_bytes: Size in bytes for each packed tensor buffer.
                          Both producer and consumer must use the same value.
        num_buffers: Number of in-flight communication slots.
    """
    if buffer_size_bytes <= 0 or num_buffers <= 0:
        raise ValueError(
            f"buffer_size_bytes and num_buffers must be positive, got "
            f"buffer_size_bytes={buffer_size_bytes}, num_buffers={num_buffers}"
        )

    def owning(tensors: list[tuple[str, torch.Tensor]]) -> None:
        restored = []
        for name, tensor in tensors:
            tensor = tensor.clone()
            # Later callbacks use the allocating unpack stream. The caller may
            # read retained clones after this helper returns.
            tensor.record_stream(caller_stream)
            restored.append((name, tensor))
        try:
            post_unpack_func(restored)
        except StopIteration as exc:
            # Preserve the existing loader-error contract.
            raise RuntimeError("post_unpack_func raised StopIteration") from exc

    def metadata():
        for name, (shape, dtype) in iterator:
            if any(dimension < 0 for dimension in shape):
                raise ValueError(f"tensor {name!r} has a negative shape: {shape}")
            yield name, (shape, dtype)

    with _hccl_group(group, num_buffers) as transport:
        caller_stream = torch.npu.current_stream()
        upstream.packed_nccl_broadcast_consumer(
            metadata(),
            transport,
            src,
            owning,
            buffer_size_bytes,
            num_buffers,
            device=f"npu:{torch.npu.current_device()}",
        )


def packed_npu_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    npu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
) -> Iterator[upstream.PackedIpcChunk]:
    """Pack tensors into one IPC buffer, refilled and yielded per chunk.

    The consumer must have read a chunk before the generator is resumed. A
    tensor is copied before the next is pulled, so the iterator may reuse one
    tensor for consecutive parameters.

    Args:
        iterator: Iterator of (name, tensor) pairs.
        npu_uuid: Physical NPU identifier of the exporting device.
        post_iter_func: Applied to each (name, tensor) before packing.
        buffer_size_bytes: Chunk size; the IPC buffer grows up to it while the
            first chunk is packed.

    """
    if buffer_size_bytes <= 0:
        raise ValueError(f"buffer_size_bytes must be positive, got {buffer_size_bytes}")
    capacity = upstream.chunk_capacity(buffer_size_bytes)
    device = torch.device("npu", torch.accelerator.current_device_index())
    post_iter_func = _checked_callback(post_iter_func)
    caller_stream = torch.npu.current_stream(device)
    buf: torch.Tensor | None = None  # at the first byte: none for no bytes
    filled = 0  # the bytes of the current chunk's copied pieces end at
    items = iter(iterator)
    # pulled to see if a chunk is last; None once `items` is exhausted
    ahead: list[tuple[str, torch.Tensor] | None] = []

    tensors: dict[int, torch.Tensor] = {}  # each tensor, until all is copied
    started: list[tuple[str, list[int], str]] = []  # since the last chunk

    def specs() -> Iterator[tuple[Sequence[int], torch.dtype]]:
        for i in itertools.count():
            item = ahead.pop() if ahead else next(items, None)
            if item is None:
                return
            t = post_iter_func(item)
            tensors[i] = t
            started.append((item[0], list(item[1].shape), upstream.dtype_name(t.dtype)))
            yield t.shape, t.dtype

    def pack(pieces: list[upstream.Piece]) -> None:
        nonlocal buf, filled
        end = upstream._bytes_end(pieces)
        with torch.npu.stream(caller_stream):
            if end:
                buf = upstream.grow_buffer(buf, end, capacity, device, filled)
                filled = end
            dsts, srcs = upstream._pack_copies(buf, pieces, tensors)
            if dsts:
                assert buf is not None
                torch._foreach_copy_(dsts, srcs)
                for src in srcs:
                    if src.device.type != "cpu":
                        src.record_stream(torch.npu.current_stream(src.device))

    for k, chunk in enumerate(upstream.chunks(specs(), buffer_size_bytes, before_pull=pack)):
        caller_stream.synchronize()  # before the consumer reads
        # the last chunk ends a tensor, and no item follows
        last = chunk.pieces[-1].position not in tensors
        if last and chunk.full:
            ahead.append(next(items, None))
            last = ahead[0] is None
        handle = None
        if k == 0 and buf is not None:  # only the first chunk grows `buf`
            handle = {npu_uuid: reduce_tensor(buf)[1]}
        yield upstream.PackedIpcChunk(
            names=[name for name, _, _ in started],
            shapes=[shape for _, shape, _ in started],
            dtype_names=[dtype for _, _, dtype in started],
            ipc_handle=handle,
            chunk_capacity=None if k else capacity,
            last=last,
        )
        filled = 0
        started.clear()


class NPUPackedIpcConsumer(upstream.PackedIpcConsumer):
    """Use the upstream IPC state machine with an NPU mapping on its first chunk."""

    def __init__(self, device_index: int, npu_uuid: str) -> None:
        if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
            raise ValueError(f"device_index must be a non-negative integer, got {device_index!r}")
        super().__init__(device_index, npu_uuid)

    def receive(self, ipc_handle, names, shapes, dtype_names, chunk_capacity, last):
        if not (len(names) == len(shapes) == len(dtype_names)):
            raise ValueError("packed IPC metadata lengths disagree")
        for name, shape, dtype_name in zip(names, shapes, dtype_names):
            if any(dimension < 0 for dimension in shape):
                raise ValueError(f"invalid metadata for tensor {name!r}")
            if not isinstance(getattr(torch, dtype_name, None), torch.dtype):
                raise ValueError(f"unknown torch dtype name {dtype_name!r}")
        if self._unpacker is None:
            if chunk_capacity is None:
                raise ValueError("chunk_capacity is required on the first packed IPC chunk")
            upstream.chunk_capacity(chunk_capacity)
            if ipc_handle:
                from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

                if self._gpu_uuid in ipc_handle and len(ipc_handle[self._gpu_uuid]) <= 6:
                    raise ValueError("NPU IPC rebuild arguments do not contain a device index")
                args = upstream.local_rebuild_args(ipc_handle, self._gpu_uuid, self._device_index)
                self._buf = rebuild_npu_tensor(*args)
            else:
                self._buf = torch.empty(0, dtype=torch.uint8, device=f"npu:{self._device_index}")
            # The upstream receiver currently hardcodes CUDA initialization.
            # Initialize only that hardware boundary, then reuse receive/close.
            self._unpacker = upstream._Unpacker(chunk_capacity, views=False)
        return super().receive(ipc_handle, names, shapes, dtype_names, chunk_capacity, last)
