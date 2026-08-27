# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.snapshot.utils import is_restore

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.distributed.parallel_state import get_mc2_group

_MAX_EXACT_ELEMENTS = 65536
_MAX_SAMPLED_ELEMENTS = 256
_MAX_PREVIEW_ELEMENTS = 16


@dataclass
class _PendingTensorTrace:
    name: str
    value: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    numel: int
    exact: bool


@dataclass
class _MoETrace:
    phase: str
    rank: int
    dp_rank: int
    tp_rank: int
    ep_rank: int
    mc2_rank: int
    layer: str
    call: int
    tensors: list[_PendingTensorTrace] = field(default_factory=list)


_CURRENT_TRACE: ContextVar[_MoETrace | None] = ContextVar("snapshot_moe_trace", default=None)


def _prefix(trace: _MoETrace) -> str:
    return (
        f"[snapshot-moe-trace] phase={trace.phase} rank={trace.rank} "
        f"dp={trace.dp_rank} tp={trace.tp_rank} ep={trace.ep_rank} "
        f"mc2={trace.mc2_rank} layer={trace.layer} call={trace.call}"
    )


def _sample_tensor(tensor: torch.Tensor, exact: bool) -> tuple[torch.Tensor, bool]:
    value = tensor.detach().reshape(-1)
    if exact and value.numel() <= _MAX_EXACT_ELEMENTS:
        return value.clone(), True
    if value.numel() <= _MAX_SAMPLED_ELEMENTS:
        return value.clone(), value.numel() == tensor.numel()
    step = max(1, value.numel() // _MAX_SAMPLED_ELEMENTS)
    return value[::step][:_MAX_SAMPLED_ELEMENTS].clone(), False


def trace_tensor(name: str, tensor: torch.Tensor | None, *, exact: bool = False) -> None:
    trace = _CURRENT_TRACE.get()
    if trace is None or tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        logger.error("%s %s has unexpected type %s", _prefix(trace), name, type(tensor).__name__)
        return
    try:
        value, is_exact = _sample_tensor(tensor, exact)
        trace.tensors.append(
            _PendingTensorTrace(
                name=name,
                value=value,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                numel=tensor.numel(),
                exact=is_exact,
            )
        )
    except Exception:
        logger.exception("%s failed to stage %s", _prefix(trace), name)


def trace_value(name: str, value: Any) -> None:
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        logger.info("%s %s=%s", _prefix(trace), name, value)


def _flush_tensor(trace: _MoETrace, pending: _PendingTensorTrace) -> None:
    value = pending.value.cpu()
    raw = value.contiguous().view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    flat = value.reshape(-1)
    preview = flat[:_MAX_PREVIEW_ELEMENTS].tolist()
    suffix = flat[-_MAX_PREVIEW_ELEMENTS:].tolist() if flat.numel() > _MAX_PREVIEW_ELEMENTS else []
    stats = ""
    if flat.numel():
        numeric = flat.to(torch.float64)
        stats = (
            f" min={numeric.min().item():.9g} max={numeric.max().item():.9g} "
            f"sum={numeric.sum().item():.17g} norm={numeric.norm().item():.17g}"
        )
    logger.info(
        "%s %s shape=%s dtype=%s numel=%s exact=%s sha256=%s%s head=%s tail=%s",
        _prefix(trace),
        pending.name,
        pending.shape,
        pending.dtype,
        pending.numel,
        pending.exact,
        digest,
        stats,
        preview,
        suffix,
    )


def _flush(trace: _MoETrace) -> None:
    if not trace.tensors:
        return
    try:
        # All fingerprints are staged on the model stream. Synchronize once,
        # after dispatch/GMM/combine have completed, before the bounded D2H copy.
        torch.npu.current_stream().synchronize()
        for pending in trace.tensors:
            _flush_tensor(trace, pending)
    except Exception:
        logger.exception("%s failed to flush tensor fingerprints", _prefix(trace))
    finally:
        trace.tensors.clear()


@contextmanager
def trace_layer(layer: str, call: int):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    trace = _MoETrace(
        phase="restore" if is_restore() else "cold",
        rank=rank,
        dp_rank=get_dp_group().rank_in_group,
        tp_rank=get_tensor_model_parallel_rank(),
        ep_rank=get_ep_group().rank_in_group,
        mc2_rank=get_mc2_group().rank_in_group,
        layer=layer,
        call=call,
    )
    token = _CURRENT_TRACE.set(trace)
    try:
        forward_context = get_forward_context()
        dp_metadata = forward_context.dp_metadata
        num_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu.tolist() if dp_metadata is not None else None
        logger.info(
            "%s begin num_tokens=%s num_tokens_across_dp=%s padded_num_tokens=%s moe_comm_type=%s",
            _prefix(trace),
            getattr(forward_context, "num_tokens", None),
            num_tokens_across_dp,
            getattr(_EXTRA_CTX, "padded_num_tokens", None),
            getattr(_EXTRA_CTX, "moe_comm_type", None),
        )
        yield
        _flush(trace)
    finally:
        _CURRENT_TRACE.reset(token)
