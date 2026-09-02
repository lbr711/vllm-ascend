# SPDX-License-Identifier: Apache-2.0

"""Persistent, derived, and transient tensor lifecycle helpers."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.logger import logger


def persist_tensor_attributes(module: nn.Module, names: Iterable[str]) -> None:
    """Convert existing tensor attributes into persistent buffers."""
    for name in names:
        tensor = getattr(module, name)
        delattr(module, name)
        module.register_buffer(name, tensor)


def persist_tensor_lists(module: nn.Module, names: Iterable[str]) -> None:
    """Register persistent aliases for tensors stored in ordinary lists."""
    for name in names:
        for index, tensor in enumerate(getattr(module, name)):
            module.register_buffer(f"_snapshot_{name}_{index}", tensor)


def set_persistent_tensor(module: nn.Module, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Create or replace a persistent buffer and return its registered tensor."""
    if name in module._buffers:
        module._buffers[name] = tensor
    else:
        module.register_buffer(name, tensor)
    return module._buffers[name]


def reset_model_modules_after_restore(models: Iterable[nn.Module | None]) -> int:
    """Reset target and drafter modules after snapshot restore.

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
            reset_state = getattr(module, "reset_snapshot_runtime_state", None)
            if callable(reset_state):
                reset_state()
                reset_count += 1
    return reset_count


def restore_derived_tensor_state(model: nn.Module, act_dtype: torch.dtype, label: str) -> None:
    restored = 0

    for module in model.modules():
        restore = getattr(module, "restore_snapshot_derived_state", None)
        if not callable(restore):
            continue
        restore(act_dtype)
        restored += 1

    logger.info(
        "[restore model] [%s] reloaded non-persistent derived weights for %d modules",
        label,
        restored,
    )
    if restored == 0:
        logger.warning(
            "[restore model] [%s] no non-persistent derived-weight reload targets found; "
            "attention decode may still use stale derived weights",
            label,
        )


def restore_global_tensor_state(
    model: nn.Module,
    hf_config: object,
    device: torch.device,
) -> None:
    from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    from vllm_ascend.ops.rotary_embedding import reload_cos_and_sin_after_restore

    restored: list[str] = []
    if AscendDSAMetadataBuilder.reload_hadamard_after_restore(hf_config, device):
        restored.append("dsa.hadamard")
    if AscendDSACPMetadataBuilder.reload_hadamard_after_restore(hf_config, device):
        restored.append("dsa_cp.hadamard")
    if AscendSFAImpl.reload_hadamard_after_restore(device):
        restored.append("sfa.hadamard")
    if reload_cos_and_sin_after_restore(model):
        restored.append("mla_rope.cos_sin")
    logger.info(
        "[restore model] rebuilt global non-persistent state: %s",
        restored if restored else "none",
    )
