# SPDX-License-Identifier: Apache-2.0

"""Dispatch snapshot-restore lifecycle hooks to model modules."""

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from vllm.logger import logger


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
    rebuilt_count = 0
    for item in _iter_modules_and_impls((model,)):
        rebuild = getattr(item, "rebuild_derived_tensors_after_snapshot_restore", None)
        if not callable(rebuild):
            continue
        rebuild(act_dtype)
        rebuilt_count += 1

    logger.info(
        "[snapshot][model] derived tensors rebuilt: model=%s hooks=%d",
        label,
        rebuilt_count,
    )
    if rebuilt_count == 0:
        logger.warning(
            "[snapshot][model] no derived-tensor rebuild hooks found; attention decode may use stale tensors: model=%s",
            label,
        )
