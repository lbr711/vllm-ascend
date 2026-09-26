# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import os
import time

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger


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


def dump_model_runner(runner, path: str = "/mnt") -> None:
    tp_size = runner.vllm_config.parallel_config.tensor_parallel_size
    model_name = runner.vllm_config.model_config.model.rstrip("/").rsplit("/", 1)[-1]
    model_dir = os.path.join(path, "snapshot_weight", f"{model_name}_dp{runner.dp_size}_tp{tp_size}")
    os.makedirs(model_dir, exist_ok=True)
    rank_in_group = get_tp_group().rank_in_group
    dump_state_dict(
        runner.get_model(),
        os.path.join(model_dir, f"model_ckpt.{runner.dp_rank}tp{rank_in_group}.pth"),
    )
    # The drafter owns a separate model and derived tensors, so it cannot share
    # the target model's checkpoint file.
    drafter_model = runner.get_draft_model()
    if drafter_model is not None:
        dump_state_dict(
            drafter_model,
            os.path.join(model_dir, f"model_ckpt_drafter.{runner.dp_rank}tp{rank_in_group}.pth"),
        )
