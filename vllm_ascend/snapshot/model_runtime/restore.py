# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group

from vllm_ascend.snapshot.model_runtime.checkpoint import dump_state_dict, restore_state_dict
from vllm_ascend.snapshot.model_runtime.module_lifecycle import (
    rebuild_model_derived_tensors_after_snapshot_restore,
    reset_modules_runtime_state,
)
from vllm_ascend.snapshot.model_runtime.tensor_lifecycle import restore_global_tensor_state
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


def get_drafter_model(runner) -> nn.Module | None:
    """Return the speculative decoder's model when it owns one."""
    if runner.vllm_config.use_v2_model_runner:
        return runner.get_draft_model()
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        return None
    model = None
    get_model = getattr(drafter, "get_model", None)
    if callable(get_model):
        try:
            model = get_model()
        except Exception:  # noqa: BLE001
            model = None
    if model is None:
        model = getattr(drafter, "model", None)
    return model if isinstance(model, nn.Module) else None


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


def _restore_model_checkpoint(runner, model: nn.Module, model_save_path: str, label: str) -> None:
    restore_state_dict(model, model_save_path, label)
    rebuild_model_derived_tensors_after_snapshot_restore(model, runner.model_config.dtype, label)


def restore_model_runner(runner, path: str = "/mnt") -> None:
    tp_size = runner.vllm_config.parallel_config.tensor_parallel_size
    model_name = runner.vllm_config.model_config.model.rstrip("/").rsplit("/", 1)[-1]
    model_dir = os.path.join(
        path,
        "snapshot_weight",
        f"{model_name}_dp{runner.dp_size}_tp{tp_size}",
    )
    rank_in_group = get_tp_group().rank_in_group
    model = runner.get_model()
    _restore_model_checkpoint(
        runner,
        model,
        os.path.join(model_dir, f"model_ckpt.{runner.dp_rank}tp{rank_in_group}.pth"),
        "model",
    )
    drafter_model = get_drafter_model(runner)
    if drafter_model is not None:
        _restore_model_checkpoint(
            runner,
            drafter_model,
            os.path.join(model_dir, f"model_ckpt_drafter.{runner.dp_rank}tp{rank_in_group}.pth"),
            "drafter",
        )

    _restore_model_runner_runtime_state(runner, model)


def _restore_model_runner_runtime_state(runner, model: nn.Module) -> None:
    """Prepare all non-checkpoint model-runner state for post-restore inference.

    This includes global derived tensors, speculative-decoding state, attention
    metadata, runner input buffers, model-module runtime state, and block tables.
    """
    restore_global_tensor_state(model, runner.model_config.hf_config, runner.device)
    if runner.vllm_config.use_v2_model_runner:
        from vllm_ascend.snapshot.model_runtime.runner_v2 import reset_runner_runtime_state

        reset_runner_runtime_state(runner)
        _reset_target_and_drafter_modules_after_restore(runner)
        return
    _reset_spec_decode_runtime_state(runner)
    _restore_drafter_runtime_state(runner)
    _reset_attention_builders_after_restore(runner)
    _rebuild_runner_native_resources(runner)
    _reset_runner_input_runtime_state(runner)
    _reset_target_and_drafter_modules_after_restore(runner)
    _reset_block_table_runtime_state(runner)


def _restore_drafter_runtime_state(runner) -> None:
    if isinstance(runner.drafter, AscendEagleProposer):
        runner.drafter.restore_runtime_buffers()


def _reset_spec_decode_runtime_state(runner) -> None:
    """Clear request carry-over owned by speculative decoding.

    The state consists of cached draft request IDs, draft token IDs and
    probabilities, and the previous request-to-batch-index mapping. The next
    request rebuilds all of them.
    """
    if hasattr(runner, "_draft_token_req_ids"):
        runner._draft_token_req_ids = None
    if hasattr(runner, "_draft_token_ids"):
        runner._draft_token_ids = None
    if hasattr(runner, "_draft_probs"):
        runner._draft_probs = None
    if hasattr(runner, "_draft_prob_req_ids"):
        runner._draft_prob_req_ids = None
    if hasattr(runner, "prev_num_spec_tokens"):
        runner.prev_num_spec_tokens = runner.num_spec_tokens
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is not None and hasattr(input_batch, "prev_req_id_to_index"):
        input_batch.prev_req_id_to_index = None


def _rebuild_runner_native_resources(runner) -> None:
    """Recreate runner-owned streams/events and discard request cache state."""
    if runner.device_metadata_executor is not None:
        from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor

        runner.device_metadata_executor = DeviceMetadataExecutor()
        assert runner.device_metadata_providers is not None
        for provider in runner.device_metadata_providers.values():
            provider.enable_device_metadata()

    runner.reset_encoder_cache()
    runner._pending_spec_decode_metadata_copies.clear()

    kvpp_scheduler = runner.kvpp.scheduler
    if kvpp_scheduler is not None:
        kvpp_scheduler._prefetch_executor.shutdown(wait=True)
        from vllm_ascend.worker.v2.kvpp import KVPPRuntime

        runner.kvpp = KVPPRuntime.create_from_kv_cache(
            vllm_config=runner.vllm_config,
            kv_cache_config=runner.kv_cache_config,
            static_forward_context=runner.compilation_config.static_forward_context,
        )


def _reset_attention_builders_after_restore(runner) -> None:
    """Reset per-iteration metadata cached by attention builders.

    Builder hooks clear request sequence lengths, block/slot mappings, context
    chunk metadata, attention-mask caches, and context-parallel staging tensors.
    """
    # Target-model metadata builders come from the KV-cache attention groups
    # constructed and owned directly by the model runner.
    builders = [
        builder
        for kv_groups in runner.attn_groups
        for attn_group in kv_groups
        for builder in attn_group.metadata_builders
    ]
    # Drafter metadata builders come from the drafter's independent attention
    # groups. DFlash, DSpark, and Step3.5 MTP all inherit AscendEagleProposer.
    if isinstance(runner.drafter, AscendEagleProposer):
        builders.extend(
            builder for attn_group in runner.drafter.draft_attn_groups for builder in attn_group.metadata_builders
        )
    # Attention-mask builders are helper objects owned by the metadata builders
    # above and carry their own per-request mask caches.
    _reset_metadata_builders(builders)


def _reset_metadata_builders(builders) -> None:
    builders_and_masks = builders + [
        builder.attn_mask_builder for builder in builders if hasattr(builder, "attn_mask_builder")
    ]
    seen_ids: set[int] = set()
    for builder_or_mask in builders_and_masks:
        if id(builder_or_mask) in seen_ids:
            continue
        seen_ids.add(id(builder_or_mask))
        reset_state = getattr(builder_or_mask, "reset_runtime_state_after_snapshot_restore", None)
        if callable(reset_state):
            reset_state()


def _reset_runner_input_runtime_state(runner) -> None:
    """Reset request and staging state owned directly by the model runner.

    The state includes Host/NPU position buffers, Host token counters, DCP
    request metadata, and the Host/NPU MoE group staging buffers.
    """
    runner.positions.zero_()
    runner._positions_cpu_buf.zero_()
    runner.input_batch.num_computed_tokens_cpu_tensor.zero_()
    runner.input_batch.num_prompt_tokens_cpu_tensor.zero_()
    if runner.use_dcp:
        runner.dcp_manager.reset_runtime_state_after_snapshot_restore()

    for staged in (runner.group_len, runner.group_key_idx, runner.group_key_cache_idx):
        staged.gpu.fill_(0)
        staged.cpu.fill_(0)


def _reset_target_and_drafter_modules_after_restore(runner) -> None:
    """Reset reusable runtime state owned by target and drafter modules.

    Modules refresh their own state and forward the hook to backend
    implementations they own.
    """
    reset_modules_runtime_state((runner.get_model(), get_drafter_model(runner)))


def _reset_block_table_runtime_state(runner) -> None:
    """Clear KV block IDs in the input batch's Host and NPU block tables.

    Graph recapture can otherwise copy snapshot-time CPU block IDs back to the
    device before a real request repopulates the active rows.
    """
    block_tables = runner.input_batch.block_table.block_tables
    for block_table in block_tables:
        buf = block_table.block_table
        buf.gpu.zero_()
        buf.cpu.zero_()
