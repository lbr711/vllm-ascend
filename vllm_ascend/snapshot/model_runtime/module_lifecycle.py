# SPDX-License-Identifier: Apache-2.0

"""Dispatch snapshot-restore lifecycle hooks to model modules."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.logger import logger


def reset_modules_runtime_state(models: Iterable[nn.Module | None]) -> int:
    """Reset runtime state held by target and drafter model modules.

    A module that owns a backend implementation is responsible for forwarding
    the hook to that implementation. Shared modules are reset only once.
    """
    reset_count = 0
    reset_ids: set[int] = set()
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if id(module) in reset_ids:
                continue
            reset_ids.add(id(module))
            reset = getattr(module, "reset_after_snapshot_restore", None)
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
    rebuilt_count = 0
    for module in model.modules():
        rebuild = getattr(module, "rebuild_derived_tensors_after_snapshot_restore", None)
        if not callable(rebuild):
            continue
        rebuild(act_dtype)
        rebuilt_count += 1

    logger.info(
        "[restore model] [%s] rebuilt non-persistent derived tensors for %d modules",
        label,
        rebuilt_count,
    )
    if rebuilt_count == 0:
        logger.warning(
            "[restore model] [%s] no derived-tensor rebuild targets found; "
            "attention decode may still use stale derived tensors",
            label,
        )
