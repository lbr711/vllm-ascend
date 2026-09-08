# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import os
import time

import torch
import torch.nn as nn
from vllm.logger import logger

from vllm_ascend.snapshot.model_runtime.h2d_copy import copy_checkpoint_tensor


def dump_state_dict(model: nn.Module, path: str) -> None:
    if os.path.exists(path):
        logger.debug(
            "[snapshot][checkpoint] dump skipped: path=%s reason=already_exists",
            path,
        )
        return

    start = time.time()
    torch.save(model.state_dict(), path)
    gc.collect()
    torch.npu.empty_cache()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception as e:
        logger.warning(
            "[snapshot][checkpoint] malloc trim failed: error=%s",
            e,
        )

    logger.info(
        "[snapshot][checkpoint] dump completed: path=%s model_type=%s duration=%.4f s",
        path,
        type(model).__name__,
        time.time() - start,
    )


def restore_state_dict(model: nn.Module, path: str, label: str) -> None:
    if not os.path.exists(path):
        logger.warning(
            "[snapshot][checkpoint] restore skipped: model=%s path=%s reason=not_found",
            label,
            path,
        )
        return

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
