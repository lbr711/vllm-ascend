# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for new restore hooks, without importing NPU-only backends.

Extract the production methods rather than duplicating their implementation.
This tests tensor contents/ownership and hook dispatch, not NPU execution.
"""

import ast
import fcntl
import io
import math
import multiprocessing
import os
import sys
import weakref
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"
HOOK = "reset_runtime_state_after_snapshot_restore"


def load_nodes(path, names, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    if methods is not None:
        for node in nodes:
            node.body = [item for item in node.body if getattr(item, "name", None) in methods]
            node.bases = [ast.Name(id="Base", ctx=ast.Load())]
            node.decorator_list = []
    tree.body = nodes
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(ROOT / path), "exec"), namespace)
    return namespace


def checkpoint_namespace():
    ns = {
        "os": os,
        "fcntl": fcntl,
        "nn": torch.nn,
        "torch": SimpleNamespace(save=torch.save, load=torch.load, npu=Mock()),
        "time": SimpleNamespace(time=lambda: 0),
        "logger": Mock(),
        "gc": SimpleNamespace(collect=lambda: None),
        "ctypes": Mock(),
        "copy_checkpoint_tensor": lambda dst, src: dst.copy_(src),
    }
    return load_nodes(
        "snapshot/model_runner_lifecycle/module_lifecycle.py", ["dump_state_dict", "restore_state_dict"], ns
    )


@pytest.mark.parametrize("snapshot_enabled", [False, True])
@pytest.mark.parametrize("is_kv_consumer", [False, True])
def test_pdmix_scale_checkpoint_preserves_storage_and_role(snapshot_enabled, is_kv_consumer):
    ns = {"nn": torch.nn, "torch": torch, "Iterable": Iterable}
    load_nodes("snapshot/model_runner_lifecycle/tensor_persistence.py", ["persist_tensor_attributes"], ns)
    ns.update(
        Base=object,
        get_current_vllm_config=lambda: SimpleNamespace(snapshot_config=object() if snapshot_enabled else None),
    )
    load_nodes(
        "quantization/methods/w8a8/w8a8_pdmix.py",
        ["AscendW8A8PDMixLinearMethod"],
        ns,
        {"process_weights_after_loading"},
    )
    scheme = ns["AscendW8A8PDMixLinearMethod"]()
    scheme._static_method = SimpleNamespace(process_weights_after_loading=Mock())
    scheme._is_kv_consumer = is_kv_consumer
    layer = torch.nn.Module()
    layer.weight_scale = torch.nn.Parameter(torch.tensor([0.5, 2.0], dtype=torch.bfloat16), requires_grad=False)
    scheme.process_weights_after_loading(layer)
    scheme._static_method.process_weights_after_loading.assert_called_once_with(layer)
    assert layer.is_kv_consumer is is_kv_consumer
    expected = layer.weight_scale.float().clone()
    torch.testing.assert_close(layer.weight_scale_fp32, expected)
    assert ("weight_scale_fp32" in layer.state_dict()) is snapshot_enabled
    assert ("weight_scale_fp32" in layer._buffers) is snapshot_enabled
    if snapshot_enabled:
        saved = io.BytesIO()
        torch.save(layer.state_dict(), saved)
        saved.seek(0)
        checkpoint = torch.load(saved, weights_only=True)
        original = layer.weight_scale_fp32
        for _ in range(2):
            original.zero_()
            layer.load_state_dict(checkpoint)
            assert layer.weight_scale_fp32 is original
            torch.testing.assert_close(layer.weight_scale_fp32, expected)


@pytest.mark.parametrize("fused_mc2", [False, True])
def test_resume_clears_force_eplb_tables_before_reuse(monkeypatch, fused_mc2):
    hook_ns = {"Base": object}
    load_nodes("ops/fused_moe/moe_comm_method.py", ["MoECommMethod"], hook_ns, {HOOK})
    base = hook_ns["MoECommMethod"]
    hook_ns["Base"] = base
    load_nodes("ops/fused_moe/moe_comm_method.py", ["FusedMC2CommImpl"], hook_ns, {HOOK})
    cls = hook_ns["FusedMC2CommImpl"] if fused_mc2 else base
    method = cls()
    method.mega_moe_symm_buffer = object()
    methods = {"active": method, "alias": method, "unused": base()}
    for module_name, attrs in (
        ("vllm.distributed.parallel_state", {"get_dp_group": Mock(), "get_ep_group": Mock()}),
        ("vllm_ascend.distributed.parallel_state", {"get_mc2_group": Mock()}),
        ("vllm_ascend.ops.fused_moe.moe_comm_method", {"_MoECommMethods": methods}),
    ):
        module = ModuleType(module_name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, module_name, module)
    ns = {
        "time": SimpleNamespace(time=lambda: 0),
        "logger": Mock(),
        "set_current_vllm_config": lambda _: nullcontext(),
        "get_distributed_init_method": lambda ip, port: f"tcp://{ip}:{port}",
    }
    load_nodes("snapshot/worker_lifecycle.py", ["_rebuild_parallel_groups"], ns)
    force_ns = {"torch": torch}
    load_nodes("ops/fused_moe/force_eplb.py", ["_build_round_robin_topk", "_build_or_get_topk"], force_ns)
    method.moe_config = SimpleNamespace(ep_size=2, ep_rank=1)
    ns["get_tp_group"] = Mock()
    worker = SimpleNamespace(
        rank=0,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_master_ip="10.0.0.1", _snapshot_data_parallel_port_list=[29502]
            )
        ),
        _init_worker_distributed_environment=Mock(),
    )
    kwargs = dict(top_k=2, num_logical_experts=8, dtype=torch.int32, device=torch.device("cpu"))
    tables = [force_ns["_build_or_get_topk"](method, num_tokens=n, **kwargs) for n in (2, 8)]
    expected = [table.clone() for table in tables]
    for table in tables:
        table.fill_(-1)  # Simulate lost device contents while Python cache keys survive.
    cache = method._force_eplb_topk_cache
    for _ in range(2):
        ns["_rebuild_parallel_groups"](worker)
        if fused_mc2:
            assert method.mega_moe_symm_buffer is None
        assert method._force_eplb_topk_cache is cache
        assert cache == {}
        for n, old, value in zip((2, 8), tables, expected):
            rebuilt = force_ns["_build_or_get_topk"](method, num_tokens=n, **kwargs)
            assert rebuilt is not old
            torch.testing.assert_close(rebuilt, value)
    assert not hasattr(methods["unused"], "_force_eplb_topk_cache")


def test_pp_checkpoint_paths_isolate_stages_but_share_between_instances():
    tp, pp = SimpleNamespace(rank_in_group=0), SimpleNamespace(rank_in_group=0)
    ns = {"os": os, "get_tp_group": lambda: tp, "get_pp_group": lambda: pp}
    load_nodes("snapshot/model_runner_lifecycle/checkpoint.py", ["model_checkpoint_path"], ns)
    config = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=2)
    runner = SimpleNamespace(
        dp_size=2,
        dp_rank=0,
        vllm_config=SimpleNamespace(parallel_config=config, model_config=SimpleNamespace(model="/models/glm/")),
    )
    paths = set()
    for dp in range(2):
        runner.dp_rank = dp
        for tp.rank_in_group in range(2):
            for pp.rank_in_group in range(2):
                path = ns["model_checkpoint_path"](runner, "/shared")
                paths.add(path)
                assert path == ns["model_checkpoint_path"](SimpleNamespace(**vars(runner)), "/shared")
                assert path != ns["model_checkpoint_path"](runner, "/shared", drafter=True)
    assert len(paths) == 8
    config.pipeline_parallel_size = 1
    assert ns["model_checkpoint_path"](runner, "/shared") == "/shared/snapshot_weight/glm_dp2_tp2/model_ckpt.1tp1.pth"


def test_pp_checkpoint_reuses_completed_shard_without_overwriting(tmp_path):
    ns = checkpoint_namespace()
    path = tmp_path / "stage.pth"
    torch.save({"weight": torch.ones(2)}, path)
    state_dict = Mock(side_effect=AssertionError("existing shard must not be overwritten"))
    ns["dump_state_dict"](SimpleNamespace(state_dict=state_dict), str(path))
    state_dict.assert_not_called()
    torch.testing.assert_close(torch.load(path)["weight"], torch.ones(2))


@pytest.mark.parametrize("partial_file", [False, True])
def test_checkpoint_serializes_processes_before_check_and_through_save(tmp_path, partial_file):
    # CPU-only, bounded fork workers; events control the race without sleeps.
    ctx = multiprocessing.get_context("fork")
    saving, blocked, release = (ctx.Event() for _ in range(3))
    path = tmp_path / "stage.pth"

    def first_writer():
        ns = checkpoint_namespace()

        def save(state, filename):
            if partial_file:
                path.write_bytes(b"incomplete checkpoint")
            saving.set()
            assert release.wait(15)
            torch.save(state, filename)

        ns["torch"].save = save
        ns["dump_state_dict"](SimpleNamespace(state_dict=lambda: {"weight": torch.ones(2)}), str(path))

    def second_writer():
        ns = checkpoint_namespace()

        def flock(fd, operation):
            assert operation == fcntl.LOCK_EX
            # The first writer must still hold the same file lock.
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, operation | fcntl.LOCK_NB)
            blocked.set()
            fcntl.flock(fd, operation)

        ns["fcntl"] = SimpleNamespace(flock=flock, LOCK_EX=fcntl.LOCK_EX)
        state_dict = Mock(side_effect=AssertionError("second writer must skip completed shard"))
        ns["dump_state_dict"](SimpleNamespace(state_dict=state_dict), str(path))
        state_dict.assert_not_called()
        torch.testing.assert_close(torch.load(path)["weight"], torch.ones(2))

    processes = []
    try:
        first = ctx.Process(target=first_writer)
        first.start()
        processes.append(first)
        assert saving.wait(15)
        second = ctx.Process(target=second_writer)
        second.start()
        processes.append(second)
        assert blocked.wait(15)
        release.set()
        for process in processes:
            process.join(15)
            assert process.exitcode == 0
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
    assert Path(f"{path}.lock").exists()


def test_checkpoint_releases_lock_on_save_failure(tmp_path):
    ns = checkpoint_namespace()
    path = tmp_path / "stage.pth"
    ns["torch"].save = Mock(side_effect=OSError("save failed"))
    with pytest.raises(OSError, match="save failed"):
        ns["dump_state_dict"](SimpleNamespace(state_dict=lambda: {}), str(path))
    with open(f"{path}.lock", "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_checkpoint_does_not_fallback_when_lock_fails(tmp_path):
    ns = checkpoint_namespace()
    ns["fcntl"] = SimpleNamespace(flock=Mock(side_effect=OSError("lock unsupported")), LOCK_EX=fcntl.LOCK_EX)
    model = SimpleNamespace(state_dict=Mock())
    with pytest.raises(OSError, match="lock unsupported"):
        ns["dump_state_dict"](model, str(tmp_path / "stage.pth"))
    model.state_dict.assert_not_called()


def pipeline_namespace(group):
    ns = {"torch": torch, "get_pp_group": lambda: group, "_refresh_pipeline_moe_groups": Mock()}
    return load_nodes(
        "snapshot/pipeline.py",
        ["prepare_pipeline_for_snapshot", "destroy_pipeline_runtime", "restore_pipeline_runtime"],
        ns,
    )


@pytest.mark.parametrize("v2", [False, True])
def test_pp_snapshot_drains_native_work_before_checkpoint(monkeypatch, v2):
    synchronize = Mock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    send, retained = Mock(), Mock()
    group = SimpleNamespace(_reap_completed_isends=Mock(), _pending_isends=deque([([retained], [object()])]))
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=0), req_states=SimpleNamespace(num_reqs=0), execute_model_state=object()
    )
    worker = SimpleNamespace(
        model_runner=runner, _pp_send_work=[send], vllm_config=SimpleNamespace(use_v2_model_runner=v2)
    )
    ns = pipeline_namespace(group)
    ns["prepare_pipeline_for_snapshot"](worker)
    send.wait.assert_called_once()
    retained.wait.assert_called_once()
    synchronize.assert_called_once()
    assert not worker._pp_send_work and not group._pending_isends
    # Non-last V2 warmup leaves execute_model_state, but no real requests.
    runner.input_batch.num_reqs = runner.req_states.num_reqs = 1
    with pytest.raises(RuntimeError, match="idle runner"):
        ns["prepare_pipeline_for_snapshot"](worker)


@pytest.mark.parametrize("v2,last", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("legacy_spec_pp", [False, True])
def test_pp_restore_resets_buffers_and_rebinds_v2_group_and_spec_closures(monkeypatch, v2, last, legacy_spec_pp):
    group = SimpleNamespace(is_last_rank=last, last_rank=3, world_size=2, make_sibling_device_group=Mock())
    ns = pipeline_namespace(group)
    protocol = ModuleType("vllm_ascend.patch.worker.patch_v2.patch_spec_pp")
    protocol._INSTALLED = "_vllm_ascend_upstream_spec_pp_installed"
    load_nodes("patch/worker/patch_v2/patch_spec_pp.py", ["install_upstream_spec_pp_protocol"], vars(protocol))
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    # Use inference tensors to catch illegal in-place reset outside inference_mode.
    with torch.inference_mode():
        hidden, residual = torch.ones(8, 4), torch.ones(8, 4)
        topk = torch.full((8, 4), 42, dtype=torch.int32)
    handler = SimpleNamespace(
        queue=deque([object()]),
        req_idx_gen_np=np.ones(2),
        device="cpu",
        main_stream=object(),
        broadcast_stream=object(),
    )
    setattr(handler, protocol._INSTALLED, True)
    old_broadcast = handler.broadcast_drafts = Mock(side_effect=AssertionError("stale closure"))
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=v2),
        intermediate_tensors=SimpleNamespace(
            tensors={"hidden_states": hidden, "residual": residual, "pp_transport_topk_indices_0": topk[:8]}
        ),
        execute_model_state=object(),
        pp_handler=handler,
        input_batch=SimpleNamespace(prev_sampled_token_ids=object()),
        kv_connector_output=object(),
        valid_sampled_token_count_gpu=object(),
        req_states=object(),
        num_speculative_steps=3,
        use_spec_pp=legacy_spec_pp,
    )
    ptrs = hidden.data_ptr(), residual.data_ptr()
    ns["restore_pipeline_runtime"](runner)
    ns["_refresh_pipeline_moe_groups"].assert_called_once_with(runner)
    assert ptrs == (hidden.data_ptr(), residual.data_ptr())
    assert not hidden.any() and not residual.any() and runner.execute_model_state is None
    assert torch.all(topk == -1)  # PP payload aliases the model-owned buffer.
    if v2:
        group.make_sibling_device_group.assert_called_once_with(group_desc="pp_broadcast")
        assert handler.broadcast_group is group.make_sibling_device_group.return_value
        assert list(handler.queue) == ([] if last else [None, None])
        assert not handler.req_idx_gen_np.any()
        assert (handler.broadcast_drafts is not old_broadcast) == legacy_spec_pp
        # A fresh closure has no previous InputBatch: no device access or broadcast.
        if last and legacy_spec_pp:
            handler.broadcast_drafts()
    else:
        group.make_sibling_device_group.assert_not_called()
        assert runner.input_batch.prev_sampled_token_ids is None
        assert runner.kv_connector_output is None and runner.valid_sampled_token_count_gpu is None


@pytest.mark.parametrize("ep_size", [1, 2])
def test_pp_refreshes_each_layers_moe_group_aliases_including_drafter(monkeypatch, ep_size):
    class MoERunner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_config = SimpleNamespace(ep_size=ep_size, tp_group=object(), dp_group=object())
            self.shared_experts = SimpleNamespace(moe_config=self.moe_config)

    tp, dp, ep, mc2 = object(), object(), object(), object()
    target, draft = torch.nn.Sequential(MoERunner(), MoERunner()), torch.nn.Sequential(MoERunner())
    imports = {
        "vllm.distributed.parallel_state": {
            "get_tp_group": lambda: tp,
            "get_dp_group": lambda: dp,
            "get_ep_group": lambda: ep,
        },
        "vllm_ascend.distributed.parallel_state": {"get_mc2_group": lambda: mc2},
        "vllm_ascend.ops.fused_moe.fused_moe": {"AscendMoERunner": MoERunner},
        "vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle": {"get_drafter_model": lambda _: draft},
    }
    for name, attributes in imports.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    ns = load_nodes("snapshot/pipeline.py", ["_refresh_pipeline_moe_groups"], {})
    ns["_refresh_pipeline_moe_groups"](SimpleNamespace(get_model=lambda: target))
    for model in (target, draft):
        for layer in model:
            config = layer.shared_experts.moe_config
            assert config is layer.moe_config
            assert config.tp_group is tp and config.dp_group is dp
            if ep_size > 1:
                assert config.ep_group is ep and config.mc2_group is mc2
            else:
                assert not hasattr(config, "ep_group") and not hasattr(config, "mc2_group")


def test_pp_v2_destroys_sibling_without_waiting_on_restored_work(monkeypatch):
    abort = ModuleType("vllm_ascend.patch.worker.patch_distributed")
    abort._abort_hccl_process_group = Mock()
    monkeypatch.setitem(sys.modules, abort.__name__, abort)
    destroy = Mock()
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    old = object()
    handler = SimpleNamespace(broadcast_group=old, queue=deque([object()]))
    work = Mock()
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        _pp_send_work=[work],
        model_runner=SimpleNamespace(pp_handler=handler),
    )
    pipeline_namespace(None)["destroy_pipeline_runtime"](worker)
    abort._abort_hccl_process_group.assert_called_once_with(old)
    destroy.assert_called_once_with(old)
    work.wait.assert_not_called()
    assert handler.broadcast_group is None and not handler.queue and not worker._pp_send_work


@pytest.mark.parametrize("pp_size", [1, 2])
def test_pp_hooks_are_snapshot_only_and_ordered(monkeypatch, pp_size):
    events = []
    pipeline = ModuleType("vllm_ascend.snapshot.pipeline")
    pipeline.prepare_pipeline_for_snapshot = lambda _: events.append("drain")
    pipeline.destroy_pipeline_runtime = lambda _: events.append("destroy_pp")
    pipeline.restore_pipeline_runtime = lambda _: events.append("restore_pp")
    monkeypatch.setitem(sys.modules, pipeline.__name__, pipeline)
    ns = {
        "time": SimpleNamespace(perf_counter=lambda: 0),
        "logger": Mock(),
        "gc": SimpleNamespace(collect=lambda: None),
        "dump_model_runner": lambda *_: events.append("dump"),
        "_call_aclrt_snapshot_api": lambda _, name: events.append(name),
        "snapshot_hccl_teardown": lambda _: nullcontext(),
        "destroy_ascend_model_parallel": lambda: events.append("destroy_ascend"),
        "cleanup_dist_env_for_snapshot": lambda: events.append("destroy_dist"),
    }
    load_nodes("snapshot/worker_lifecycle.py", ["suspend_worker", "_parallel_group_cleanup", "_run_timed_steps"], ns)
    worker = SimpleNamespace(
        rank=0,
        model_runner=object(),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size), snapshot_config=object()
        ),
    )
    ns["suspend_worker"](worker)
    assert events == (["drain"] if pp_size > 1 else []) + [
        "dump",
        "aclrtSnapShotProcessLock",
        "aclrtSnapShotProcessBackup",
    ]
    events.clear()
    ns["_parallel_group_cleanup"](worker)
    assert events == (["destroy_pp"] if pp_size > 1 else []) + ["destroy_ascend", "destroy_dist"]
    events.clear()
    ns.update(
        {
            "reload_cos_and_sin_after_restore": lambda _: None,
            "reset_moe_padding_cache_after_snapshot_restore": lambda: None,
            "_reset_v1_runner_runtime_state": lambda _: events.append("runner"),
            "_reset_target_and_drafter_modules_after_restore": lambda _: events.append("modules"),
        }
    )
    load_nodes("snapshot/model_runner_lifecycle/restore.py", ["_restore_model_runner_runtime_state"], ns)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=worker.vllm_config.parallel_config, use_v2_model_runner=False),
        get_model=lambda: object(),
    )
    ns["_restore_model_runner_runtime_state"](runner)
    assert events == ["runner", "modules"] + (["restore_pp"] if pp_size > 1 else [])


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("ip", ["10.0.0.2", "2001:db8::2", "", "0.0.0.0", "::"])
def test_resume_resolves_worker_ip_for_all_parallel_configurations(pp_size, ip):
    ns = {
        name: Mock()
        for name in (
            "_call_aclrt_snapshot_api",
            "_reset_triton_kernel_caches",
            "_update_worker_info",
            "_parallel_group_cleanup",
            "_prepare_kv_transfer_for_snapshot_restore",
            "_rebuild_parallel_groups",
            "restore_model_runner",
            "_recapture_graph",
            "_rebuild_kv_transfer_engine",
        )
    }
    ns.update({"get_ip": Mock(return_value=ip), "time": SimpleNamespace(perf_counter=lambda: 0), "logger": Mock()})
    load_nodes("snapshot/worker_lifecycle.py", ["resume_worker", "_run_timed_steps"], ns)
    worker = SimpleNamespace(
        rank=0,
        model_runner=object(),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)),
    )
    if ip in ("", "0.0.0.0", "::"):
        with pytest.raises(RuntimeError, match="local worker IP"):
            ns["resume_worker"](worker, "master", "/shared", "new-id")
        ns["_call_aclrt_snapshot_api"].assert_not_called()
        ns["_rebuild_kv_transfer_engine"].assert_not_called()
    else:
        ns["resume_worker"](worker, "master", "/shared", "new-id")
        ns["_update_worker_info"].assert_called_once_with(worker, ip, "master")
        ns["_rebuild_kv_transfer_engine"].assert_called_once_with(worker, ip, "new-id")
    ns["get_ip"].assert_called_once_with(force=True)


@pytest.mark.parametrize("last", [False, True])
def test_pp_target_and_last_stage_drafter_checkpoint_roundtrip(tmp_path, last):
    ns = checkpoint_namespace()
    model, draft = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2) if last else None
    ns.update(
        {
            "get_tp_group": lambda: SimpleNamespace(rank_in_group=0),
            "get_pp_group": lambda: SimpleNamespace(rank_in_group=int(last)),
            "get_drafter_model": lambda runner: draft,
            "_restore_model_runner_runtime_state": Mock(),
            "rebuild_model_derived_tensors_after_snapshot_restore": Mock(),
        }
    )
    load_nodes("snapshot/model_runner_lifecycle/checkpoint.py", ["model_checkpoint_path", "dump_model_runner"], ns)
    load_nodes("snapshot/model_runner_lifecycle/restore.py", ["restore_model_runner", "_restore_model_checkpoint"], ns)
    runner = SimpleNamespace(
        dp_size=1,
        dp_rank=0,
        get_model=lambda: model,
        model_config=SimpleNamespace(dtype=torch.float32),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=2),
            model_config=SimpleNamespace(model="glm"),
        ),
    )
    originals = [m.weight.clone() for m in (model, draft) if m is not None]
    ns["dump_model_runner"](runner, str(tmp_path))
    for m in (model, draft):
        if m is not None:
            m.weight.data.zero_()
    ns["restore_model_runner"](runner, str(tmp_path))
    for m, weight in zip((m for m in (model, draft) if m is not None), originals):
        torch.testing.assert_close(m.weight, weight)
    assert len(list(tmp_path.rglob("*.pth"))) == 1 + int(last)


def test_worker_snapshot_entrypoints_delegate_with_lazy_imports(monkeypatch):
    module_name = "vllm_ascend.snapshot.worker_lifecycle"
    lifecycle = ModuleType(module_name)
    lifecycle.suspend_worker = Mock()
    lifecycle.unlock_worker = Mock()
    lifecycle.resume_worker = Mock()
    monkeypatch.setitem(sys.modules, module_name, lifecycle)
    ns = {"Base": object}
    load_nodes("worker/worker.py", ["NPUWorker"], ns, {"suspend", "device_unlock", "resume"})
    worker = ns["NPUWorker"]()
    worker.suspend("/tmp/model")
    worker.device_unlock()
    worker.resume("master-ip", "/tmp/model", "new-id")
    lifecycle.suspend_worker.assert_called_once_with(worker, "/tmp/model")
    lifecycle.unlock_worker.assert_called_once_with(worker)
    lifecycle.resume_worker.assert_called_once_with(worker, "master-ip", "/tmp/model", "new-id")


@pytest.mark.parametrize("fail_stage", [None, "destroy_groups", "destroy_kv"])
def test_resume_destroys_groups_then_kv_before_rebuilding(fail_stage):
    events = []

    def cleanup(_worker):
        events.append("destroy_groups")
        if fail_stage == "destroy_groups":
            raise RuntimeError("group teardown failed")

    def prepare(_worker):
        events.append("destroy_kv")
        if fail_stage == "destroy_kv":
            raise RuntimeError("transport teardown failed")

    ns = {
        "get_ip": lambda **_: "new-ip",
        "_call_aclrt_snapshot_api": lambda _, name: events.append(name),
        "_reset_triton_kernel_caches": lambda: None,
        "_update_worker_info": lambda *_: None,
        "_parallel_group_cleanup": cleanup,
        "_prepare_kv_transfer_for_snapshot_restore": prepare,
        "_rebuild_parallel_groups": lambda _: events.append("groups"),
        "restore_model_runner": lambda *_: events.append("model"),
        "_recapture_graph": lambda _: events.append("graph"),
        "_rebuild_kv_transfer_engine": lambda *_: events.append("create_kv"),
        "time": SimpleNamespace(perf_counter=lambda: 0),
        "logger": Mock(),
    }
    load_nodes("snapshot/worker_lifecycle.py", ["resume_worker", "_run_timed_steps"], ns)
    worker = SimpleNamespace(
        rank=0,
        model_runner=object(),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1)),
    )
    if fail_stage is not None:
        with pytest.raises(RuntimeError, match="teardown failed"):
            ns["resume_worker"](worker, "master-ip")
        expected = ["aclrtSnapShotProcessRestore", "aclrtSnapShotProcessUnlock", "destroy_groups"]
        if fail_stage == "destroy_kv":
            expected.append("destroy_kv")
        assert events == expected
    else:
        ns["resume_worker"](worker, "master-ip")
        assert events == [
            "aclrtSnapShotProcessRestore",
            "aclrtSnapShotProcessUnlock",
            "destroy_groups",
            "destroy_kv",
            "groups",
            "model",
            "graph",
            "create_kv",
        ]


@pytest.mark.parametrize("role,has_group", [(None, True), ("none", True), ("producer", False), ("consumer", True)])
def test_snapshot_kv_prepare_respects_configuration(role, has_group):
    group = Mock()
    ns = {"has_kv_transfer_group": lambda: has_group, "get_kv_transfer_group": Mock(return_value=group)}
    load_nodes("snapshot/worker_lifecycle.py", ["_prepare_kv_transfer_for_snapshot_restore"], ns)
    config = (
        None if role is None else SimpleNamespace(is_kv_producer=role == "producer", is_kv_consumer=role == "consumer")
    )
    ns["_prepare_kv_transfer_for_snapshot_restore"](
        SimpleNamespace(vllm_config=SimpleNamespace(kv_transfer_config=config))
    )
    assert group.prepare_for_snapshot_restore.call_count == int(role == "consumer" and has_group)


def make_snapshot_kv_worker(kind, producer):
    """Load actual transport lifecycle methods without importing NPU backends."""
    engine = Mock()
    engine.unregister_memory.return_value = 0
    new_engine = Mock()
    new_engine.register_memory.return_value = 0
    transfer_engine_manager = SimpleNamespace(transfer_engine=engine, hostname="old-ip", register_buffer=Mock())

    def reset():
        transfer_engine_manager.transfer_engine = None
        transfer_engine_manager.hostname = None

    def create(hostname, device_name):
        transfer_engine_manager.transfer_engine = new_engine
        transfer_engine_manager.hostname = hostname
        return new_engine

    transfer_engine_manager.reset = Mock(side_effect=reset)
    transfer_engine_manager.get_transfer_engine = Mock(side_effect=create)
    new_thread = Mock()
    ns = {
        "Base": object,
        "global_te": transfer_engine_manager,
        "logger": Mock(),
        "get_tp_group": Mock(),
        "threading": SimpleNamespace(Event=Mock),
        "KVCacheSendingThread": Mock(return_value=new_thread),
        "KVCacheRecvingLayerThread": Mock(return_value=new_thread),
        "MooncakeAgentMetadata": Mock(),
    }
    cls = "MooncakeLayerwiseConnectorWorker" if kind == "layerwise" else "MooncakeConnectorWorker"
    filename = "mooncake_connector.py" if kind == "normal" else f"mooncake_{kind}_connector.py"
    load_nodes(
        f"distributed/kv_transfer/kv_p2p/{filename}",
        [cls],
        ns,
        {"prepare_for_snapshot_restore", "rebuild_kv_transfer_endpoint"},
    )
    worker = ns[cls]()
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=producer, is_kv_consumer=not producer)
    )
    worker.engine = engine
    worker._registered_regions = ([4096], [8192]) if kind == "hybrid" else SimpleNamespace(ptrs=[4096], lengths=[8192])
    worker._engine_device_name = "npu"
    worker._sync_engine_id_after_snapshot = Mock()
    worker.tp_rank = worker.pcp_rank = 0
    worker.tp_size = worker.pd_head_ratio = worker._prefill_tp_size = 1
    worker.engine_id, worker.side_channel_port = "engine", 12345
    worker.kv_caches = {}
    worker.xfer_handshake_metadata = Mock()
    worker.layer_metadata = []
    listener = Mock()
    listener.is_alive.return_value = False
    has_peer = producer if kind == "layerwise" else not producer
    peer = SimpleNamespace(engine=engine if has_peer else None)
    if kind == "layerwise":
        worker.kv_recv_layer_thread = None if producer else listener
        worker.kv_send_layer_thread = peer if producer else None
        worker.k_buffer = worker.v_buffer = None
    else:
        worker.kv_send_thread = listener if producer else None
        worker.kv_recv_thread = None if producer else peer
    return worker, transfer_engine_manager, listener, peer


@pytest.mark.parametrize("kind", ["normal", "hybrid", "layerwise"])
@pytest.mark.parametrize("producer", [False, True])
def test_snapshot_kv_prepare_drops_last_engine_reference(kind, producer):
    worker, transfer_engine_manager, listener, peer = make_snapshot_kv_worker(kind, producer)
    old_engine = weakref.ref(worker.engine)
    worker.prepare_for_snapshot_restore()
    worker.prepare_for_snapshot_restore()
    assert old_engine() is None
    assert worker.engine is None
    transfer_engine_manager.get_transfer_engine.assert_not_called()
    transfer_engine_manager.reset.assert_called_once()
    worker.rebuild_kv_transfer_endpoint("new-ip", "new-engine")
    transfer_engine_manager.reset.assert_called_once()  # rebuild must not tear down again
    transfer_engine_manager.get_transfer_engine.assert_called_once()
    transfer_engine_manager.register_buffer.assert_called_once_with([4096], [8192])
    has_listener = (not producer) if kind == "layerwise" else producer
    assert listener.stop.call_count == int(has_listener)
    if not has_listener:
        assert peer.engine is worker.engine


@pytest.mark.parametrize("failure", ["listener", "unregister"])
def test_snapshot_kv_prepare_failure_does_not_create_transport(failure):
    worker, transfer_engine_manager, listener, _ = make_snapshot_kv_worker("normal", True)
    listener.is_alive.return_value = failure == "listener"
    if failure == "unregister":
        worker.engine.unregister_memory.return_value = -1
    with pytest.raises(RuntimeError):
        worker.prepare_for_snapshot_restore()
    transfer_engine_manager.reset.assert_not_called()
    transfer_engine_manager.get_transfer_engine.assert_not_called()


@pytest.mark.parametrize("mode", ["shared", "independent", "fabric"])
@pytest.mark.parametrize("lazy,initialized", [(False, True), (True, True), (True, False)])
def test_snapshot_pool_prepare_preserves_engine_mode_and_lazy_init(mode, lazy, initialized):
    _, transfer_engine_manager, _, _ = make_snapshot_kv_worker("normal", False)
    config = SimpleNamespace(protocol="ascend")
    ns = {
        "Base": object,
        "global_te": transfer_engine_manager,
        "MooncakeStoreConfig": SimpleNamespace(load_from_env=lambda: config),
    }
    load_nodes(
        "distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py",
        ["MooncakeBackend"],
        ns,
        {"prepare_for_snapshot_restore", "reset_after_snapshot"},
    )
    pool = ns["MooncakeBackend"]()
    pool._store_was_initialized = None
    pool._store_initialized, pool._lazy_init = initialized, lazy
    pool._use_fabric_mem, pool._use_store_independent_te = mode == "fabric", mode == "independent"
    pool._registered_buffers = ([4096], [8192]) if initialized else None
    pool.store = object() if initialized else None
    pool._setup_store, pool.register_buffer = Mock(), Mock()
    pool.prepare_for_snapshot_restore()
    assert pool.store is None
    assert transfer_engine_manager.reset.call_count == int(mode == "shared")
    pool._setup_store.assert_not_called()
    # A P2P sibling may now create the replacement singleton, even on the same IP.
    replacement = transfer_engine_manager.get_transfer_engine("old-ip", None)
    pool.reset_after_snapshot("old-ip")
    assert transfer_engine_manager.transfer_engine is replacement
    assert transfer_engine_manager.reset.call_count == int(mode == "shared")
    assert pool._setup_store.call_count == int(initialized or not lazy)
    assert pool.register_buffer.call_count == int(initialized)
    assert pool._store_was_initialized is None


@pytest.mark.parametrize("pool_first", [False, True])
def test_snapshot_multi_connector_releases_shared_engine_then_reuses_replacement(pool_first):
    worker, transfer_engine_manager, _, peer = make_snapshot_kv_worker("normal", False)
    old_engine = weakref.ref(worker.engine)
    unregister = worker.engine.unregister_memory
    config = SimpleNamespace(protocol="ascend")
    ns = {
        "Base": object,
        "global_te": transfer_engine_manager,
        "MooncakeStoreConfig": SimpleNamespace(load_from_env=lambda: config),
    }
    load_nodes(
        "distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py",
        ["MooncakeBackend"],
        ns,
        {"prepare_for_snapshot_restore", "reset_after_snapshot", "register_buffer"},
    )
    pool = ns["MooncakeBackend"]()
    pool._store_was_initialized = None
    pool._store_initialized = True
    pool._lazy_init = pool._use_fabric_mem = pool._use_store_independent_te = False
    pool._registered_buffers = ([4096], [8192])
    pool.store = SimpleNamespace(engine=worker.engine)
    pool._setup_store = lambda: SimpleNamespace(
        engine=transfer_engine_manager.get_transfer_engine(pool._local_hostname, None)
    )

    class PoolConnector:
        def prepare_for_snapshot_restore(self):
            pool.prepare_for_snapshot_restore()

        def rebuild_kv_transfer_endpoint(self, local_ip, new_engine_id):
            pool.reset_after_snapshot(local_ip)

    ns = {"Base": object, "AscendStoreConnector": PoolConnector}
    load_nodes(
        "distributed/kv_transfer/ascend_multi_connector.py",
        ["AscendMultiConnector"],
        ns,
        {"prepare_for_snapshot_restore", "rebuild_kv_transfer_endpoint"},
    )
    connector = ns["AscendMultiConnector"]()
    connector._connectors = [PoolConnector(), worker] if pool_first else [worker, PoolConnector()]
    connector.prepare_for_snapshot_restore()
    unregister.assert_called_once_with(4096)  # shared regions must not be unregistered twice
    del unregister  # Mock bound-method ownership must not hold the old engine alive
    import gc

    gc.collect()
    assert old_engine() is None
    transfer_engine_manager.get_transfer_engine.assert_not_called()
    resets = transfer_engine_manager.reset.call_count
    connector.rebuild_kv_transfer_endpoint("old-ip", "new-id")
    assert transfer_engine_manager.reset.call_count == resets
    assert worker.engine is transfer_engine_manager.transfer_engine is pool.store.engine is peer.engine


@pytest.mark.parametrize("legacy", [False, True])
def test_turboquant_restore_preserves_contents_and_addresses(legacy):
    path = ROOT / "quantization/methods/kv_cache/turboquant/latent.py"
    tree = ast.parse(path.read_text())
    centroids = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "CENTROIDS")
    samples = next(
        n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "_LEGACY_LLOYD_MAX_SAMPLE_COUNT"
    )
    ns = {
        "torch": torch,
        "np": np,
        "math": math,
        "cache": cache,
        "HEAD_DIM": 512,
        "CENTROIDS": ast.literal_eval(centroids.value),
        "_LEGACY_LLOYD_MAX_SAMPLE_COUNT": ast.literal_eval(samples.value),
    }
    load_nodes(str(path.relative_to(ROOT)), ["TurboQuantLatent", "_get_legacy_transforms"], ns)
    latent = ns["TurboQuantLatent"](legacy_hadamard=legacy)
    getattr(latent, HOOK)()
    assert latent.rotation is None
    latent._initialize(torch.device("cpu"))
    names = ("rotation", "centroids", "norm_lut") + (("inverse_rotation",) if legacy else ())
    sibling = ns["TurboQuantLatent"](legacy_hadamard=legacy)
    sibling._initialize(torch.device("cpu"))
    original = {name: getattr(latent, name).clone() for name in names}
    addresses = {name: getattr(latent, name).data_ptr() for name in names}
    for _ in range(2):
        for name in names:
            getattr(latent, name).fill_(42)
        getattr(latent, HOOK)()
        for name in names:
            torch.testing.assert_close(getattr(latent, name), original[name], rtol=0, atol=0)
            assert getattr(latent, name).data_ptr() == addresses[name]
    if legacy:
        # Both already-initialized and later-created layers use the same cache.
        later = ns["TurboQuantLatent"](legacy_hadamard=True)
        later._initialize(torch.device("cpu"))
        for owner in (sibling, later):
            for name in names:
                assert getattr(owner, name) is getattr(latent, name)
                torch.testing.assert_close(getattr(owner, name), original[name], rtol=0, atol=0)


@pytest.mark.parametrize("c8,c4", [(False, False), (True, False), (False, True)])
def test_indexer_restores_hadamard_for_both_quantization_paths(c8, c4):
    ns = {"Base": object, "torch": torch}
    load_nodes(
        "attention/indexer.py",
        ["AscendSFAIndexerBackend"],
        ns,
        {"enable_sparse_li_quant", "rebuild_derived_tensors_after_snapshot_restore"},
    )
    backend = ns["AscendSFAIndexerBackend"]()
    backend.enable_sparse_li_c8, backend.enable_sparse_li_c4 = c8, c4
    old = backend.q_hadamard = backend.k_hadamard = torch.zeros(2)

    def rebuild():
        assert backend.q_hadamard is None and backend.k_hadamard is None
        backend.q_hadamard = backend.k_hadamard = torch.ones(2)

    backend.process_weights_after_loading = Mock(side_effect=rebuild)
    for _ in range(2):
        backend.rebuild_derived_tensors_after_snapshot_restore(torch.bfloat16)
    assert backend.process_weights_after_loading.call_count == (2 if c8 or c4 else 0)
    if not (c8 or c4):
        assert backend.q_hadamard is old


@pytest.mark.parametrize("enabled", [False, True])
def test_sfa_restores_nested_turboquant_helper(enabled):
    ns = {"Base": object, "PreprocessType": SimpleNamespace(PROLOG_V3=object())}
    load_nodes("attention/sfa_v1.py", ["AscendSFAImpl"], ns, {HOOK})
    impl = ns["AscendSFAImpl"]()
    impl.topk_indices_buffer = torch.zeros(2, dtype=torch.int32)
    impl.turboquant = Mock() if enabled else None
    impl.preprocess_type = None
    getattr(impl, HOOK)()
    assert torch.all(impl.topk_indices_buffer == -1)
    if enabled:
        impl.turboquant.reset_runtime_state_after_snapshot_restore.assert_called_once()


def test_gqa_pcp_builder_rebinds_group_and_clears_batch_references():
    group = object()
    ns = {"Base": object, "get_pcp_group": lambda: group}
    load_nodes("attention/context_parallel/attention_cp.py", ["AscendAttentionDCPMetadataBuilder"], ns, {HOOK})
    builder = ns["AscendAttentionDCPMetadataBuilder"]()
    builder.pcp_group = builder._pcp_context = builder._pcp_cache_group_idx = object()
    for _ in range(2):
        getattr(builder, HOOK)()
        assert builder.pcp_group is group
        assert builder._pcp_context is None and builder._pcp_cache_group_idx is None


@pytest.mark.parametrize("optional_buffers", [False, True])
def test_v41_builder_refills_metadata_and_rebinds_source_rope(optional_buffers):
    rope = (torch.ones(2), torch.zeros(2))
    ns = {"Base": object, "get_full_cos_and_sin_dsa_for_layer": lambda name: rope}
    load_nodes("attention/dsa_v41.py", ["AscendDSAV41MetadataBuilder"], ns, {HOOK, "prepare_source_rope"})
    builder = ns["AscendDSAV41MetadataBuilder"]()
    neg_names = ("_slot_mapping", "_slot_mapping_2d", "_flat_slot_mapping")
    zero_names = (
        "_seq_lens",
        "_cache_seq_lens",
        "_cmp_residual",
        "_smla_metadata",
        "_qli_metadata",
        "_c2_ring_metadata",
        "_c2_complete_mask",
        "_c2_source_positions",
        "_c2_source_sin",
    )
    optional_neg = ("_dspark_swa_indices", "_a5_causal_swa_indices")
    optional_zero = ("_dspark_swa_lengths", "_a5_causal_swa_lengths", "_a5_smla_metadata", "_a5_smla_length_rows")
    for name in (*neg_names, *zero_names, "_c2_source_cos"):
        setattr(builder, name, torch.full((4,), 42))
    for name in (*optional_neg, *optional_zero):
        setattr(builder, name, torch.full((4,), 42) if optional_buffers else None)
    builder._device_metadata_tasks = (object(),)
    builder._device_metadata_enabled = True
    builder._build_compressor_metadata = True
    builder._cache_kind = "compressor_state"
    builder._c2_rope_layer_names = ["layer0"]
    builder._c2_full_source_rope = object()
    for _ in range(2):
        getattr(builder, HOOK)()
        for name in (*neg_names, *optional_neg):
            value = getattr(builder, name)
            assert value is None or torch.all(value == -1)
        for name in (*zero_names, *optional_zero):
            value = getattr(builder, name)
            assert value is None or torch.all(value == 0)
        assert torch.all(builder._c2_source_cos == 1)
        assert builder._device_metadata_tasks == () and builder._device_metadata_enabled
        assert builder._c2_full_source_rope is rope


def test_v41_pcp_rebuilds_rope_aliases_before_first_metadata_build():
    ns = {"Base": type("Base", (), {HOOK: lambda self: self.prepare_source_rope()})}
    load_nodes("attention/context_parallel/dsa_v41_cp.py", ["AscendDSAV41PCPMetadataBuilder"], ns, {HOOK})
    builder = ns["AscendDSAV41PCPMetadataBuilder"]()
    builder._hidden_restore_idx_buffer = torch.ones(2, dtype=torch.int64)
    builder._pcp_rope_buffers = {"old": object()}

    def prepare():
        assert not builder._pcp_rope_buffers
        assert torch.all(builder._hidden_restore_idx_buffer == 0)
        builder._pcp_rope_buffers["fresh"] = object()

    builder.prepare_source_rope = Mock(side_effect=prepare)
    for _ in range(2):
        getattr(builder, HOOK)()
        assert set(builder._pcp_rope_buffers) == {"fresh"}


def test_kpool_discards_metadata_and_query_caches():
    ns = {"Base": object}
    load_nodes("attention/indexer_kpool.py", ["AscendIndexerKPoolMetadataBuilder"], ns, {HOOK})
    builder = ns["AscendIndexerKPoolMetadataBuilder"]()
    builder._metadata_buffers = {1: object()}
    builder._dsa_cp_query_buffers = {2: torch.ones(2)}
    for _ in range(2):
        getattr(builder, HOOK)()
        assert not builder._metadata_buffers and not builder._dsa_cp_query_buffers


def test_multimodal_sentinel_cache_is_invalidated_without_changing_parameters():
    ns = {"Base": object}
    load_nodes("models/deepseek_v4/vl_model.py", ["AscendDeepseekV4ForConditionalGeneration"], ns, {HOOK})
    model = ns["AscendDeepseekV4ForConditionalGeneration"]()
    model._sentinel_table = torch.zeros(4, 2)
    parameter = model.image_start = torch.ones(2)
    for _ in range(2):
        getattr(model, HOOK)()
        assert model._sentinel_table is None
        assert model.image_start is parameter and torch.all(parameter == 1)


@pytest.mark.parametrize("enabled,initialized", [(False, False), (True, False), (True, True)])
def test_pcp_weight_switch_rebinds_group_and_rebuilds_copies(enabled, initialized):
    base_reset = Mock()
    base = type("Base", (), {HOOK: lambda self: base_reset()})
    group, config = object(), object()
    get_group = Mock(return_value=group)
    from_group = Mock(return_value=config)
    ns = {"Base": base, "get_pcp_group": get_group, "WeightSwitchConfig": SimpleNamespace(from_group=from_group)}
    load_nodes("attention/context_parallel/dsa_cp.py", ["AscendDSAPCPImpl"], ns, {HOOK})
    impl = ns["AscendDSAPCPImpl"]()
    impl.enable_pcp_o_proj_weight_sharding = enabled
    impl._pcp_o_proj_use_full_weight = True
    states = [Mock(), Mock()]
    impl._pcp_o_proj_weight_switches = [(None, None, state) for state in states] if initialized else None
    getattr(impl, HOOK)()
    base_reset.assert_called_once()
    assert impl._pcp_o_proj_use_full_weight is False
    if enabled:
        from_group.assert_called_once_with(group)
        assert impl.pcp_o_proj_weight_switch_config is config
    else:
        get_group.assert_not_called()
    for state in states:
        if enabled and initialized:
            state.rebuild_after_snapshot_restore.assert_called_once_with(config)
        else:
            state.rebuild_after_snapshot_restore.assert_not_called()


def test_pcp_builder_resets_nested_global_builder():
    base_reset = Mock()
    ns = {"Base": type("Base", (), {HOOK: lambda self: base_reset()})}
    load_nodes("attention/context_parallel/dsa_cp.py", ["AscendDSAPCPMetadataBuilder"], ns, {HOOK})
    builder = ns["AscendDSAPCPMetadataBuilder"]()
    builder._global_metadata_builder = Mock()
    getattr(builder, HOOK)()
    base_reset.assert_called_once()
    builder._global_metadata_builder.reset_runtime_state_after_snapshot_restore.assert_called_once()


@pytest.mark.parametrize("initialized", [False, True])
def test_mla_current_kv_index_constant_restored_in_place(initialized):
    ns = {"Base": object, "torch": torch}
    load_nodes("attention/context_parallel/mla_cp.py", ["AscendMlaDCPImpl"], ns, {HOOK})
    impl = ns["AscendMlaDCPImpl"]()
    impl._refresh_dcp_group = Mock()
    indices = torch.full((8,), 123, dtype=torch.int64)
    address = indices.data_ptr()
    scratch = torch.full((8,), 456)
    impl._dcp_current_kv_buffers = (scratch, scratch, indices) if initialized else None
    for _ in range(2):
        getattr(impl, HOOK)()
        if initialized:
            torch.testing.assert_close(indices, torch.arange(8))
            assert indices.data_ptr() == address
        assert (scratch == 456).all()  # Overwritten by prolog, not a constant.
    assert impl._refresh_dcp_group.call_count == 2


def test_dsa_builder_invalidates_turboquant_geometry():
    ns = {"Base": object}
    load_nodes("attention/dsa_v1.py", ["AscendDSAMetadataBuilder"], ns, {HOOK})
    builder = ns["AscendDSAMetadataBuilder"]()
    for name in (
        "start_pos_prefill",
        "sas_metadata_buffer",
        "qli_metadata_buffer",
        "qli_seqused_k",
        "qli_cmp_residual_k",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "seqused_q",
        "_zero_i32",
        "slot_mapping",
    ):
        setattr(builder, name, torch.ones(2, dtype=torch.int32))
    builder.compressor_metadata_buffers = None
    builder.common_ratio_to_sas_metadata = {}
    builder.dspark_swa_indices_buffer = None
    builder.hadamard = None
    builder.tq_group_block_sizes = torch.tensor([128, 256])
    getattr(builder, HOOK)()
    assert builder.tq_group_block_sizes is None


def test_dsa_impl_restores_turboquant_only_when_present():
    ns = {"Base": object}
    load_nodes("attention/dsa_v1.py", ["AscendDSAImpl"], ns, {HOOK})
    impl = ns["AscendDSAImpl"]()
    impl.turboquant = Mock()
    getattr(impl, HOOK)()
    impl.turboquant.reset_runtime_state_after_snapshot_restore.assert_called_once()
    impl.turboquant = None
    getattr(impl, HOOK)()


def test_module_dispatch_refills_mxfp_scale_cache_once(monkeypatch):
    path = "vllm_ascend.attention.attention_c8_mxfp"
    backend = ModuleType(path)
    backend.AscendC8MXFPAttentionBackendImpl = type("Impl", (), {})
    fill_ns = {"torch": torch}
    load_nodes("attention/attention_c8_mxfp.py", ["fill_mxfp_v_scale_cache"], fill_ns)
    backend.fill_mxfp_v_scale_cache = Mock(wraps=fill_ns["fill_mxfp_v_scale_cache"])
    monkeypatch.setitem(sys.modules, path, backend)
    ns = {"torch": torch, "nn": torch.nn, "Iterable": Iterable, "Iterator": Iterator}
    load_nodes(
        "snapshot/model_runner_lifecycle/module_lifecycle.py",
        ["_iter_modules_and_impls", "reset_modules_runtime_state"],
        ns,
    )
    model = torch.nn.Module()
    model.impl = backend.AscendC8MXFPAttentionBackendImpl()
    model.v_cache_scale = torch.arange(16, dtype=torch.uint8)
    with torch.inference_mode():
        cache = torch.zeros((3, 1, 1, 2, 16, 2), dtype=torch.uint8)
    address = cache.data_ptr()
    model.kv_cache = (None, None, None, cache)
    ns["reset_modules_runtime_state"]((model, model, None))
    backend.fill_mxfp_v_scale_cache.assert_called_once_with(model.v_cache_scale, model.kv_cache[3])
    torch.testing.assert_close(cache, model.v_cache_scale.view(1, 1, 1, 1, 16, 1).expand_as(cache))
    assert cache.data_ptr() == address
    backend.fill_mxfp_v_scale_cache.reset_mock()
    model.impl = object()
    ns["reset_modules_runtime_state"]((model,))
    backend.fill_mxfp_v_scale_cache.assert_not_called()
