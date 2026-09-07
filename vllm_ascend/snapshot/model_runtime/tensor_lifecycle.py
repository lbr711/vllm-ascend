# SPDX-License-Identifier: Apache-2.0

"""Model tensor persistence and global tensor restoration helpers."""

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
        "[snapshot][model] global tensors rebuilt: tensors=%s",
        ",".join(restored) if restored else "none",
    )
