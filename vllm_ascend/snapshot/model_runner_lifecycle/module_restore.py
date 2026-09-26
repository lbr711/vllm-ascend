# SPDX-License-Identifier: Apache-2.0

"""Restore model module tensors and runtime state."""

import os
import time
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from vllm.logger import logger

from vllm_ascend.snapshot.model_runner_lifecycle.h2d_copy import copy_checkpoint_tensor


def restore_state_dict(model: nn.Module, path: str, label: str) -> None:
    if not os.path.exists(path):
        logger.error(
            "[snapshot][checkpoint] restore failed: model=%s path=%s reason=not_found",
            label,
            path,
        )
        raise FileNotFoundError(f"Snapshot checkpoint does not exist: {path}")

    start = time.time()
    state_dict = torch.load(path, map_location="cpu", mmap=True)
    restored = 0
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name, cpu_tensor in state_dict.items():
        if name in parameters:
            copy_checkpoint_tensor(parameters[name].data, cpu_tensor)
            restored += 1
        if name in buffers:
            copy_checkpoint_tensor(buffers[name].data, cpu_tensor)
            restored += 1
    logger.info(
        "[snapshot][checkpoint] restore completed: model=%s path=%s tensors=%d/%d duration=%.4f s",
        label,
        path,
        restored,
        len(state_dict),
        time.time() - start,
    )


def _iter_modules_and_impls(models: Iterable[nn.Module | None]) -> Iterator[object]:
    """Yield each model module followed by its backend implementation."""
    visited_ids: set[int] = set()
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            for item in (module, getattr(module, "impl", None)):
                if item is None or id(item) in visited_ids:
                    continue
                visited_ids.add(id(item))
                yield item


def reset_modules_runtime_state(models: Iterable[nn.Module | None]) -> int:
    """Reset runtime state held by target and drafter model modules.

    Hooks on a module and its backend implementation are dispatched centrally.
    Shared objects are reset only once.
    """
    reset_count = 0
    for item in _iter_modules_and_impls(models):
        reset = getattr(item, "reset_runtime_state_after_snapshot_restore", None)
        if callable(reset):
            reset()
            reset_count += 1
    return reset_count


def rebuild_model_derived_tensors_after_snapshot_restore(
    model: nn.Module,
    act_dtype: torch.dtype,
    label: str,
) -> None:
    """Rebuild non-persistent derived tensors through model module hooks."""
    for item in _iter_modules_and_impls((model,)):
        rebuild = getattr(item, "rebuild_derived_tensors_after_snapshot_restore", None)
        if not callable(rebuild):
            continue
        rebuild(act_dtype)
