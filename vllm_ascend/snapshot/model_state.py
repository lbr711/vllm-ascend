# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import os
import time

import torch
import torch.nn as nn
import torch_npu
from vllm.logger import logger

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def dump_state_dict(model: nn.Module, path: str) -> None:
    if os.path.exists(path):
        logger.info("model save path %s exists, skip dump model", path)
        return

    logger.info("[dump model] start dump model to %s (type=%s)", path, type(model))
    start = time.time()
    import psutil  # type: ignore[import-untyped]

    process = psutil.Process(os.getpid())
    logger.info("start dump_model() cpu memory use: %.2f MB", process.memory_info().rss / 1024**2)
    torch.save(model.state_dict(), path)
    gc.collect()
    logger.info("after gc.collect() cpu memory use: %.2f MB", process.memory_info().rss / 1024**2)
    torch.npu.empty_cache()
    logger.info("after torch.npu.empty_cache() cpu memory use: %.2f MB", process.memory_info().rss / 1024**2)
    try:
        libc = ctypes.CDLL("libc.so.6")
        result = libc.malloc_trim(0)
        if result == 1:
            print("exec malloc_trim(0) success")
        else:
            print("exec malloc_trim(0) fail")
    except Exception as e:
        print(f"exec malloc_trim(0) with error: {e}")

    logger.info("after dump_model() cpu memory use: %.2f MB", process.memory_info().rss / 1024**2)
    logger.info("[dump model] save model ckpt to %s, elapse %.4f s", path, time.time() - start)


def _is_nz_int32_packed_weight(tensor: torch.Tensor) -> bool:
    return (
        tensor.dtype == torch.int32
        and tensor.device.type == "npu"
        and int(torch_npu.get_npu_format(tensor)) == ACL_FORMAT_FRACTAL_NZ
    )


def _copy_into_nz_int32_weight(dst: torch.Tensor, cpu_tensor: torch.Tensor) -> None:
    """Restore the cold-start int8-NZ -> int32-view packed representation."""
    if cpu_tensor.dtype != torch.int32:
        raise RuntimeError(f"NZ int32 packed weight restore expects int32 cpu tensor, got {cpu_tensor.dtype}")
    if cpu_tensor.shape != dst.shape:
        raise RuntimeError(
            f"NZ int32 packed weight shape mismatch: cpu {tuple(cpu_tensor.shape)} vs dst {tuple(dst.shape)}"
        )

    # Cold start casts int8 to NZ before viewing the storage as packed int32.
    cpu_i8 = cpu_tensor.contiguous().view(torch.int8)
    tmp = torch.empty(cpu_i8.shape, dtype=torch.int8, device=dst.device)
    tmp.copy_(cpu_i8)
    tmp = torch_npu.npu_format_cast(tmp, ACL_FORMAT_FRACTAL_NZ)
    dst.view(torch.int8).copy_(tmp)


def _restore_tensor(dst: torch.Tensor, cpu_tensor: torch.Tensor) -> None:
    if _is_nz_int32_packed_weight(dst):
        _copy_into_nz_int32_weight(dst, cpu_tensor)
    else:
        dst.copy_(cpu_tensor)


def restore_state_dict(model: nn.Module, path: str, label: str) -> None:
    if not os.path.exists(path):
        logger.warning("[restore model] [%s] ckpt %s not found, skip", label, path)
        return

    start = time.time()
    state_dict = torch.load(path, map_location="cpu", mmap=True)
    logger.info(
        "[restore model] [%s] load model to cpu from %s, elapse %ss, the num of items is %s",
        label,
        path,
        time.time() - start,
        len(state_dict),
    )
    restored = 0
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name, cpu_tensor in state_dict.items():
        if name in parameters:
            _restore_tensor(parameters[name].data, cpu_tensor)
            restored += 1
        if name in buffers:
            _restore_tensor(buffers[name].data, cpu_tensor)
            restored += 1
    logger.info("[restore model] [%s] replace success %s / %s", label, restored, len(state_dict))
    logger.info(
        "[restore model] [%s] restore model ckpt from %s, elapse %.4f s",
        label,
        path,
        time.time() - start,
    )
