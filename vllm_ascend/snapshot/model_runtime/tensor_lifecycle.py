# SPDX-License-Identifier: Apache-2.0

"""Model tensor persistence and global tensor restoration helpers."""

from collections.abc import Iterable

import torch
import torch.nn as nn


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


def restore_global_tensor_state(
    model: nn.Module,
    hf_config: object,
    device: torch.device,
) -> None:
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
    from vllm_ascend.ops.rotary_embedding import reload_cos_and_sin_after_restore

    if AscendSFAIndexerBackend.q_hadamard is not None:
        import scipy.linalg

        hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device=device) / (128**0.5)
        AscendSFAIndexerBackend.q_hadamard = hadamard
        AscendSFAIndexerBackend.k_hadamard = hadamard.clone()
    reload_cos_and_sin_after_restore(model)
