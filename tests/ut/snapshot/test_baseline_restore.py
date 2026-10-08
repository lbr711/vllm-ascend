# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for new restore hooks, without importing NPU-only backends.

Extract the production methods rather than duplicating their implementation.
This tests tensor contents/ownership and hook dispatch, not NPU execution.
"""

import ast
import math
import sys
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
