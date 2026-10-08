# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for new restore hooks, without importing NPU-only backends.

Extract the production methods rather than duplicating their implementation.
This tests tensor contents/ownership and hook dispatch, not NPU execution.
"""

import ast
import math
import sys
import weakref
from collections.abc import Iterable, Iterator
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


@pytest.mark.parametrize("fail_prepare", [False, True])
def test_resume_releases_kv_transport_before_groups(fail_prepare):
    events = []

    def prepare(_worker):
        events.append("destroy_kv")
        if fail_prepare:
            raise RuntimeError("transport teardown failed")

    ns = {
        "_call_aclrt_snapshot_api": lambda _, name: events.append(name),
        "_reset_triton_kernel_caches": lambda: None,
        "_update_worker_info": lambda *_: None,
        "_prepare_kv_transfer_for_snapshot_restore": prepare,
        "_rebuild_parallel_groups": lambda _: events.append("groups"),
        "restore_model_runner": lambda *_: events.append("model"),
        "_recapture_graph": lambda _: events.append("graph"),
        "_rebuild_kv_transfer_engine": lambda *_: events.append("create_kv"),
        "time": SimpleNamespace(perf_counter=lambda: 0),
        "logger": Mock(),
    }
    load_nodes("snapshot/worker_lifecycle.py", ["resume_worker", "_run_timed_steps"], ns)
    worker = SimpleNamespace(rank=0, model_runner=object())
    if fail_prepare:
        with pytest.raises(RuntimeError, match="transport teardown failed"):
            ns["resume_worker"](worker, "new-ip", "master-ip")
        assert events == ["aclrtSnapShotProcessRestore", "aclrtSnapShotProcessUnlock", "destroy_kv"]
    else:
        ns["resume_worker"](worker, "new-ip", "master-ip")
        assert events == [
            "aclrtSnapShotProcessRestore",
            "aclrtSnapShotProcessUnlock",
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


def test_turboquant_restore_preserves_contents_and_addresses():
    path = ROOT / "quantization/methods/kv_cache/turboquant/latent.py"
    tree = ast.parse(path.read_text())
    centroids = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "CENTROIDS")
    ns = {"torch": torch, "np": np, "math": math, "HEAD_DIM": 512, "CENTROIDS": ast.literal_eval(centroids.value)}
    load_nodes(str(path.relative_to(ROOT)), ["TurboQuantLatent"], ns)
    latent = ns["TurboQuantLatent"]()
    getattr(latent, HOOK)()
    assert latent.rotation is None
    latent._initialize(torch.device("cpu"))
    names = ("rotation", "centroids", "norm_lut")
    original = {name: getattr(latent, name).clone() for name in names}
    addresses = {name: getattr(latent, name).data_ptr() for name in names}
    for _ in range(2):
        for name in names:
            getattr(latent, name).fill_(42)
        getattr(latent, HOOK)()
        for name in names:
            torch.testing.assert_close(getattr(latent, name), original[name], rtol=0, atol=0)
            assert getattr(latent, name).data_ptr() == addresses[name]


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
