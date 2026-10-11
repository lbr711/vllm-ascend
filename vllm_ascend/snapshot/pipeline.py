# SPDX-License-Identifier: Apache-2.0
"""Snapshot-only lifecycle for pipeline-parallel work and buffers."""

import torch
from vllm.distributed.parallel_state import get_pp_group


def prepare_pipeline_for_snapshot(worker) -> None:
    runner = worker.model_runner
    state = runner.req_states if worker.vllm_config.use_v2_model_runner else runner.input_batch
    if state.num_reqs:
        raise RuntimeError("PP snapshot requires an idle runner with no active requests")
    # V2 non-last stages retain execute_model_state after dummy runs. That is
    # not an active request; it is discarded with the other restore-time state.

    # Complete native Work objects before checkpointing, never wait on restored Work.
    for handle in worker._pp_send_work:
        handle.wait()
    worker._pp_send_work.clear()
    group = get_pp_group()
    group._reap_completed_isends()
    for handles, _ in group._pending_isends:
        for handle in handles:
            handle.wait()
    torch.npu.synchronize()
    group._pending_isends.clear()


def destroy_pipeline_runtime(worker) -> None:
    worker._pp_send_work.clear()
    if worker.vllm_config.use_v2_model_runner:
        from vllm_ascend.patch.worker.patch_distributed import _abort_hccl_process_group

        handler = worker.model_runner.pp_handler
        # This sibling group is not owned by the global PP GroupCoordinator.
        _abort_hccl_process_group(handler.broadcast_group)
        torch.distributed.destroy_process_group(handler.broadcast_group)
        handler.broadcast_group = None
        handler.queue.clear()


@torch.inference_mode()
def restore_pipeline_runtime(runner) -> None:
    _refresh_pipeline_moe_groups(runner)
    runner.execute_model_state = None
    # Dummy runs read these buffers before any receive fills them. Preserve
    # storage and all auxiliary payload keys for graph capture and real receives.
    if runner.intermediate_tensors is not None:
        for tensor in runner.intermediate_tensors.tensors.values():
            tensor.zero_()

    if not runner.vllm_config.use_v2_model_runner:
        runner.input_batch.prev_sampled_token_ids = None
        runner.kv_connector_output = None
        runner.valid_sampled_token_count_gpu = None
        return

    handler = runner.pp_handler
    group = get_pp_group()
    handler.broadcast_group = group.make_sibling_device_group(group_desc="pp_broadcast")
    handler.is_last_rank = group.is_last_rank
    handler.last_rank = group.last_rank
    handler.req_idx_gen_np.fill(0)
    handler.queue.clear()
    if not handler.is_last_rank:
        handler.queue.extend([None] * group.world_size)
    if runner.use_spec_pp:
        from vllm_ascend.patch.worker.patch_v2.patch_spec_pp import (
            _INSTALLED,
            install_upstream_spec_pp_protocol,
        )

        # Rebind the closures to discard their captured previous InputBatch.
        setattr(handler, _INSTALLED, False)
        install_upstream_spec_pp_protocol(handler, runner.req_states, runner.num_speculative_steps)


def _refresh_pipeline_moe_groups(runner) -> None:
    from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group

    from vllm_ascend.distributed.parallel_state import get_mc2_group
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import get_drafter_model

    # The global comm-method registry retains only the last config of each
    # execution shape. Earlier layers' shared experts still own their configs.
    for model in (runner.get_model(), get_drafter_model(runner)):
        if model is None:
            continue
        for module in model.modules():
            if isinstance(module, AscendMoERunner):
                config = module.moe_config
                config.tp_group = get_tp_group()
                config.dp_group = get_dp_group()
                if config.ep_size > 1:
                    config.ep_group = get_ep_group()
                    config.mc2_group = get_mc2_group()
