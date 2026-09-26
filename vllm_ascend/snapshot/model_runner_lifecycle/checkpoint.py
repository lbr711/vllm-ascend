# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import os
import time

import torch
import torch.nn as nn
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
