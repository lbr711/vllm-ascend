# SPDX-License-Identifier: Apache-2.0

import os

from vllm.distributed.parallel_state import get_tp_group

from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import (
    dump_state_dict,
    get_drafter_model,
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
    drafter_model = get_drafter_model(runner)
    if drafter_model is not None:
        dump_state_dict(
            drafter_model,
            os.path.join(model_dir, f"model_ckpt_drafter.{runner.dp_rank}tp{rank_in_group}.pth"),
        )
