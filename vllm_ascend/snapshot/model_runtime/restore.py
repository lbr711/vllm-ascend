# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group

from vllm_ascend.snapshot.model_runtime.checkpoint import dump_state_dict, restore_state_dict
from vllm_ascend.snapshot.model_runtime.module_lifecycle import (
    rebuild_model_derived_tensors_after_snapshot_restore,
    reset_modules_runtime_state,
)
from vllm_ascend.snapshot.model_runtime.tensor_lifecycle import restore_global_tensor_state


def get_drafter_model(runner) -> nn.Module | None:
    """Return the speculative decoder's model when it owns one."""
    return runner.get_draft_model()


def _require_model_runner_v2(runner) -> None:
    if not runner.vllm_config.use_v2_model_runner:
        raise RuntimeError("Snapshot lifecycle requires Model Runner V2")


def dump_model_runner(runner, path: str = "/mnt") -> None:
    _require_model_runner_v2(runner)
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
    _require_model_runner_v2(runner)
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
    restore_global_tensor_state(model)
    _reset_runner_runtime_state(runner)
    _reset_target_and_drafter_modules_after_restore(runner)


def _reset_input_buffers(buffers) -> None:
    for name in (
        "input_ids",
        "positions",
        "is_padding",
        "query_start_loc",
        "seq_lens",
        "dcp_local_seq_lens",
    ):
        getattr(buffers, name).zero_()


def _reset_staged_tensor(buffer) -> None:
    buffer.gpu.zero_()
    buffer.clear_staged_writes()


def _reset_request_state(state) -> None:
    # Snapshot creation is an idle-service operation, not request migration.
    # Preserve the RequestState object: sampler/PCP helpers also reference it.
    if state.num_reqs:
        raise RuntimeError("Snapshot restore requires an empty Model Runner V2 request state")
    for name in ("all_token_ids", "total_len", "num_computed_tokens"):
        _reset_staged_tensor(getattr(state, name))
    for name in ("prompt_len", "prefill_len"):
        buffer = getattr(state, name)
        buffer.cpu.zero_()
        buffer.copy_to_uva()
    state.num_computed_prefill_tokens.fill(0)
    state.num_computed_tokens_np.fill(0)
    state.max_seq_len.fill(0)
    state.last_sampled_tokens.zero_()
    state.draft_tokens.zero_()
    state.next_prefill_tokens.zero_()


def _reset_block_tables(tables) -> None:
    for table in tables.block_tables:
        _reset_staged_tensor(table)
    tables.num_blocks.cpu.zero_()
    tables.num_blocks.copy_to_uva()
    for table in tables.input_block_tables:
        table.zero_()
    tables.slot_mappings.fill_(-1)
    # These tensors contain device addresses and layout constants, not request
    # data. Rebuild them instead of zeroing them like the block IDs above.
    tables.init_block_table_layout_tensors()


def _reset_speculator(speculator) -> None:
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    _reset_input_buffers(speculator.input_buffers)
    for name in ("idx_mapping", "temperature", "seeds", "draft_tokens", "hidden_states"):
        getattr(speculator, name).zero_()
    if speculator.draft_logits is not None:
        _, fill = speculator.draft_logits_spec(speculator.vllm_config)
        speculator.draft_logits.fill_(fill)
    speculator.input_batch = None
    if isinstance(speculator, AutoRegressiveSpeculator):
        speculator.input_buffers.seq_lens_cpu.zero_()
        speculator.current_draft_step.zero_()
        speculator.last_token_indices.zero_()
        speculator.sample_src_positions.zero_()
        if speculator.inputs_embeds is not None:
            speculator.inputs_embeds.zero_()
    if isinstance(speculator, DFlashSpeculator):
        speculator.context_positions.zero_()
        speculator.sample_indices.zero_()
        speculator.sample_pos.zero_()
        speculator.sample_idx_mapping.fill_(-1)
        speculator.sample_col.copy_(
            torch.arange(
                speculator.num_speculative_steps,
                dtype=torch.int32,
                device=speculator.device,
            ).repeat(speculator.max_num_reqs)
        )
        speculator._context_slot_mappings.fill_(-1)
    if isinstance(speculator, DFlash2Speculator):
        speculator._anchor_indices.copy_(
            torch.arange(
                speculator.max_num_reqs,
                dtype=torch.int64,
                device=speculator.device,
            )
            * speculator.num_query_per_req
        )
        speculator._selector_scores.zero_()
        speculator._cached_candidate_ids.zero_()
    if isinstance(speculator, DSparkSpeculator):
        speculator._step_cols.copy_(
            torch.arange(
                speculator.num_speculative_steps,
                dtype=torch.int32,
                device=speculator.device,
            )
        )
        speculator._anchor_idx.copy_(
            torch.arange(
                speculator.max_num_reqs,
                dtype=torch.int64,
                device=speculator.device,
            )
            * speculator.num_query_per_req
        )
        speculator.draft_token_confidence_probs.zero_()
        if speculator._d2t_scatter_index is not None:
            d2t = speculator.model.draft_id_to_target_id
            speculator._d2t_scatter_index.copy_(torch.arange(d2t.shape[0], device=d2t.device) + d2t)
            speculator._draft_scatter_buf.fill_(float("-inf"))


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


def _reset_runner_runtime_state(runner) -> None:
    from vllm.v1.worker.gpu import pcp_manager
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    from vllm_ascend.worker.v2.kvpp import KVPPRuntime

    runner.execute_model_state = None
    _reset_request_state(runner.req_states)
    _reset_input_buffers(runner.input_buffers)
    runner.input_buffers.seq_lens_cpu.zero_()
    runner.num_computed_tokens_cpu.zero_()
    _reset_block_tables(runner.block_tables)
    runner.reset_encoder_cache()
    runner.draft_tokens_handler.req_ids = []
    runner.draft_tokens_handler.draft_tokens_np = None
    runner.draft_tokens_handler.num_draft_tokens = 0

    # PCP stores local block-table pointer arrays and per-batch restore indices.
    # Reuse its cold-start constructor, then reconnect the shared consumers.
    runner.pcp_manager = pcp_manager.maybe_build_pcp_manager(
        runner.vllm_config,
        runner.device,
        runner.supports_mm_inputs,
        runner.req_states,
        runner.block_tables,
        cls=runner.pcp_manager_cls,
    )
    if runner.pcp_manager is not None:
        runner.pcp_manager.vllm_config = runner.vllm_config
        runner.model_state.pcp_manager = runner.pcp_manager
        if runner.speculator is not None:
            runner.speculator.pcp_manager = runner.pcp_manager

    groups = list(runner.attn_groups)
    if isinstance(runner.speculator, DraftModelSpeculator):
        _reset_speculator(runner.speculator)
        groups.extend(runner.speculator.attn_groups)
    _reset_metadata_builders(
        [builder for kv_groups in groups for group in kv_groups for builder in group.metadata_builders]
    )

    if runner.kvpp.scheduler is not None:
        runner.kvpp.scheduler._prefetch_executor.shutdown(wait=True)
        runner.kvpp = KVPPRuntime.create_from_kv_cache(
            vllm_config=runner.vllm_config,
            kv_cache_config=runner.kv_cache_config,
            static_forward_context=runner.compilation_config.static_forward_context,
        )
        runner.model_state.kvpp_runtime = runner.kvpp


def _reset_target_and_drafter_modules_after_restore(runner) -> None:
    """Reset reusable runtime state owned by target and drafter modules.

    Modules refresh their own state and forward the hook to backend
    implementations they own.
    """
    reset_modules_runtime_state((runner.get_model(), get_drafter_model(runner)))


def reset_graph_managers(runner) -> None:
    """Drop Model Runner V2 graph handles after the ACL Graph pool reset."""
    from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

    manager = runner.cudagraph_manager
    runner.cudagraph_manager = ModelAclGraphManager(
        runner.vllm_config,
        runner.device,
        manager.cudagraph_mode,
        runner.decode_query_len,
        runner,
        lora_capture_cases=manager.lora_capture_cases,
        varlen_decode=manager.varlen_decode,
    )
    if runner.speculator is not None:
        runner.speculator.init_cudagraph_manager(manager.cudagraph_mode)
