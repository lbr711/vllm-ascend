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
        logger.info(
            "[snapshot][checkpoint] dump skipped: path=%s reason=already_exists",
            path,
        )
        return

    logger.info(
        "[snapshot][checkpoint] dump started: path=%s model_type=%s",
        path,
        type(model).__name__,
    )
    start = time.time()
    import psutil  # type: ignore[import-untyped]

    process = psutil.Process(os.getpid())
    logger.info(
        "[snapshot][checkpoint] memory usage: phase=before_dump rss=%.2f MiB",
        process.memory_info().rss / 1024**2,
    )
    torch.save(model.state_dict(), path)
    gc.collect()
    logger.info(
        "[snapshot][checkpoint] memory usage: phase=after_gc rss=%.2f MiB",
        process.memory_info().rss / 1024**2,
    )
    torch.npu.empty_cache()
    logger.info(
        "[snapshot][checkpoint] memory usage: "
        "phase=after_npu_cache_clear rss=%.2f MiB",
        process.memory_info().rss / 1024**2,
    )
    try:
        libc = ctypes.CDLL("libc.so.6")
        result = libc.malloc_trim(0)
        logger.info(
            "[snapshot][checkpoint] malloc trim completed: released=%s",
            result == 1,
        )
    except Exception as e:
        logger.warning(
            "[snapshot][checkpoint] malloc trim failed: error=%s",
            e,
        )

    logger.info(
        "[snapshot][checkpoint] memory usage: phase=after_dump rss=%.2f MiB",
        process.memory_info().rss / 1024**2,
    )
    logger.info(
        "[snapshot][checkpoint] dump completed: path=%s duration=%.4f s",
        path,
        time.time() - start,
    )


def restore_state_dict(model: nn.Module, path: str, label: str) -> None:
    if not os.path.exists(path):
        logger.warning(
            "[snapshot][checkpoint] restore skipped: "
            "model=%s path=%s reason=not_found",
            label,
            path,
        )
        return

    start = time.time()
    state_dict = torch.load(path, map_location="cpu", mmap=True)
    logger.info(
        "[snapshot][checkpoint] checkpoint loaded: "
        "model=%s path=%s tensors=%d duration=%.4f s",
        label,
        path,
        len(state_dict),
        time.time() - start,
    )
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
        "[snapshot][checkpoint] tensors copied: model=%s restored=%d total=%d",
        label,
        restored,
        len(state_dict),
    )
    logger.info(
        "[snapshot][checkpoint] restore completed: "
        "model=%s path=%s duration=%.4f s",
        label,
        path,
        time.time() - start,
    )
