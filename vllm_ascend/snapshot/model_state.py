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
    """True for NZ + int32-packed quant weights (e.g. W4A8 after pack_to_int32)."""
    if tensor.dtype != torch.int32 or tensor.device.type != "npu":
        return False
    return int(torch_npu.get_npu_format(tensor)) == ACL_FORMAT_FRACTAL_NZ


def _copy_into_nz_int32_weight(dst: torch.Tensor, cpu_tensor: torch.Tensor) -> None:
    """Write CPU int32-packed weights into an existing NZ int32 Parameter.

    Cold-start does NZ cast on int8, then view(int32). TransData cannot NZ-cast
    int32 directly, so restore mirrors that order on a temporary NPU buffer and
    writes back through an int8 view of dst to preserve the original storage.
    """
    cpu = cpu_tensor.detach().contiguous()
    if cpu.dtype != torch.int32:
        raise RuntimeError(
            f"NZ int32 packed weight restore expects int32 cpu tensor, got {cpu.dtype}"
        )
    if cpu.shape != dst.shape:
        raise RuntimeError(
            f"NZ int32 packed weight shape mismatch: cpu {tuple(cpu.shape)} vs dst {tuple(dst.shape)}"
        )

    # int32 packed storage <-> int8 layout used by npu_format_cast (last dim * 4).
    cpu_i8 = cpu.view(torch.int8)
    tmp = torch.empty(cpu_i8.shape, dtype=torch.int8, device=dst.device)
    tmp.copy_(cpu_i8)
    tmp = torch_npu.npu_format_cast(tmp, ACL_FORMAT_FRACTAL_NZ)

    dst_i8 = dst.view(torch.int8)
    if dst_i8.shape != tmp.shape:
        raise RuntimeError(
            f"NZ int32 packed weight int8-view shape mismatch: "
            f"dst_i8 {tuple(dst_i8.shape)} vs tmp {tuple(tmp.shape)}"
        )
    dst_i8.copy_(tmp)
    del tmp, cpu_i8, dst_i8


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
