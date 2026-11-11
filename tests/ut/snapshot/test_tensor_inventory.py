# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for generic static tensor discovery and in-place restore."""

import ast
import ctypes
import fcntl
import gc
import importlib.util
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "tensor_inventory", ROOT / "vllm_ascend/snapshot/model_runner_lifecycle/tensor_inventory.py"
)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


def model_with_tensors():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(3))
    model.register_buffer("persistent", torch.ones(3))
    model.register_buffer("temporary_constant", torch.arange(3), persistent=False)
    model._derived = torch.arange(6.0).reshape(2, 3)
    model.nested = {"weights": [model._derived, (model._derived.t(),)]}
    model.impl = SimpleNamespace(_scale=torch.ones(2))
    model.kv_cache = [torch.empty(0)]
    model.cycle = []
    model.cycle.append(model.cycle)
    return model


def test_inventory_covers_private_containers_impl_and_aliases(tmp_path):
    model = model_with_tensors()
    inventory.capture_tensor_inventory(model)
    tensors = inventory.inventory_tensors(model)
    assert len(tensors) == 5
    assert all(t is not model.weight and t is not model.persistent for t in tensors.values())
    assert any(t is model.temporary_constant for t in tensors.values())
    path = tmp_path / "inventory.pth"
    torch.save(inventory.inventory_state(model), path)
    state = torch.load(path, mmap=True, weights_only=True)
    assert len(state["aliases"]) == 1
    pointers = {p: (id(t), t.data_ptr()) for p, t in tensors.items()}
    for t in tensors.values():
        t.zero_()
    with torch.inference_mode():
        for _ in range(2):
            for p, t in inventory.validate_inventory(model, state).items():
                t.copy_(state["tensors"][p])
    assert torch.equal(model._derived, torch.arange(6.0).reshape(2, 3))
    assert model.nested["weights"][0] is model._derived
    assert pointers == {p: (id(t), t.data_ptr()) for p, t in inventory.inventory_tensors(model).items()}


def test_inventory_boundary_excludes_later_runtime_and_empty_kv():
    model = model_with_tensors()
    inventory.capture_tensor_inventory(model)
    paths = model._snapshot_tensor_paths
    model.kv_cache[0] = torch.ones(128)
    model.request_state = torch.ones(128)
    assert model._snapshot_tensor_paths == paths
    assert all(t.numel() != 128 for t in inventory.inventory_tensors(model).values())


@pytest.mark.parametrize("change", ["shape", "dtype", "alias", "removed", "version"])
def test_inventory_rejects_incompatible_state(change):
    model = model_with_tensors()
    inventory.capture_tensor_inventory(model)
    state = inventory.inventory_state(model)
    if change == "shape":
        model.impl._scale = torch.ones(4)
    elif change == "dtype":
        model.impl._scale = torch.ones(2, dtype=torch.int32)
    elif change == "alias":
        model.nested["weights"][0] = model._derived.clone()
    elif change == "removed":
        model.impl._scale = None
    else:
        state["version"] = 99
    with pytest.raises((TypeError, ValueError)):
        inventory.validate_inventory(model, state)


def test_inventory_does_not_evaluate_properties():
    class Model(torch.nn.Module):
        @property
        def dangerous(self):
            raise AssertionError("property evaluated")

    model = Model()
    inventory.capture_tensor_inventory(model)
    assert inventory.inventory_tensors(model) == {}


def checkpoint_namespace(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.snapshot.model_runner_lifecycle.tensor_inventory", inventory)
    monkeypatch.setattr(torch, "npu", Mock(), raising=False)
    source = ROOT / "vllm_ascend/snapshot/model_runner_lifecycle/module_lifecycle.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if getattr(node, "name", "") in ("dump_state_dict", "restore_state_dict")]
    copies = []

    def copy(dst, src):
        copies.append(dst.data_ptr())
        dst.copy_(src)

    namespace = dict(
        torch=torch,
        nn=torch.nn,
        os=os,
        fcntl=fcntl,
        gc=gc,
        ctypes=ctypes,
        time=time,
        logger=Mock(),
        copy_checkpoint_tensor=copy,
    )
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace, copies


def test_checkpoint_dispatch_roundtrip_and_no_legacy_reuse(tmp_path, monkeypatch):
    namespace, copies = checkpoint_namespace(monkeypatch)
    with torch.inference_mode():
        model = model_with_tensors()
    inventory.capture_tensor_inventory(model)
    path = str(tmp_path / "model.pth")
    namespace["dump_state_dict"](model, path)
    assert os.path.exists(path)
    assert "inventory" in torch.load(path, weights_only=True)
    before = {p: (id(t), t.data_ptr()) for p, t in inventory.inventory_tensors(model).items()}
    with torch.inference_mode():
        for t in inventory.inventory_tensors(model).values():
            t.zero_()
    namespace["restore_state_dict"](model, path, "test")
    assert torch.equal(model._derived, torch.arange(6.0).reshape(2, 3))
    assert before == {p: (id(t), t.data_ptr()) for p, t in inventory.inventory_tensors(model).items()}
    assert model._derived.data_ptr() in copies
    # An old-format file must not be silently reused or overwritten.
    torch.save({"old": torch.ones(1)}, path)
    with pytest.raises(ValueError, match="lacks tensor inventory"):
        namespace["dump_state_dict"](model, path)
    with pytest.raises(ValueError, match="lacks tensor inventory"):
        namespace["restore_state_dict"](model, path, "test")
    assert "old" in torch.load(path, weights_only=True)
    os.unlink(path)
    with pytest.raises(FileNotFoundError):
        namespace["restore_state_dict"](model, path, "test")


@pytest.mark.parametrize("kind", ["weightless_norm", "kpool", "mla", "sfa"])
def test_inventory_replaces_removed_model_hooks(tmp_path, monkeypatch, kind):
    namespace, _ = checkpoint_namespace(monkeypatch)
    model = torch.nn.Module()
    if kind == "weightless_norm":
        model.weight = torch.ones(8)
        original = {"weight": model.weight.clone()}
        owner = model
    elif kind == "kpool":
        source = ROOT / "vllm_ascend/attention/indexer_kpool.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextKPoolIndexerBackend")
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading"
        )
        ns = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
        model.head_dim = 2
        model.wk_weights_proj = torch.nn.Linear(3, 4, bias=False)
        model.index_kpool_compress_gate = torch.nn.Parameter(torch.ones(3))
        model.k_norm = torch.nn.LayerNorm(3)
        ns["process_weights_after_loading"](model)
        names = ("_key_weight", "_gate_weight", "_head_weight_f32", "_norm_weight_f32", "_norm_bias_f32")
        original = {name: getattr(model, name).clone() for name in names}
        owner = model
    else:
        names = [
            "W_UV",
            "W_UK_T",
            "weight_dq",
            "weight_dkv_kr",
            "weight_uq_qr",
            "dequant_scale_w_uq_qr",
            "dequant_scale_w_dq",
            "dequant_scale_w_dkv_kr",
        ]
        if kind == "mla":
            names += ["mlapo_W_UK_T"]
        else:
            names += [
                "weight_dq_scale",
                "weight_dkv_kr_scale",
                "weight_uq_qr_scale",
                "wd_qkv",
                "deq_scale_qkv",
                "quant_bias_qkv",
                "wu_q",
                "qb_deq_scl",
                "qb_qt_bias",
                "ctkv_scale",
                "q_nope_scale",
            ]
        owner = model.impl = SimpleNamespace(**{name: torch.randn(4) for name in names})
        if kind == "mla":
            indices = torch.arange(4)
            owner._strided_kv_workspace = (torch.zeros(4), torch.zeros(4), indices)
            owner._flashmla_writer_buffers = (torch.zeros(4), torch.zeros(4), indices)
            names += ["indices"]
            owner.indices = indices  # Keep an external consumer alias as well.
        original = {name: getattr(owner, name).clone() for name in names}
    inventory.capture_tensor_inventory(model)
    addresses = {name: (id(getattr(owner, name)), getattr(owner, name).data_ptr()) for name in original}
    path = str(tmp_path / "model.pth")
    namespace["dump_state_dict"](model, path)
    for _ in range(2):
        with torch.no_grad():
            for name in original:
                getattr(owner, name).zero_()
        namespace["restore_state_dict"](model, path, kind)
        for name, expected in original.items():
            actual = getattr(owner, name)
            torch.testing.assert_close(actual, expected)
            assert (id(actual), actual.data_ptr()) == addresses[name]


@pytest.mark.parametrize(
    "file,cls,method",
    [
        ("ops/layernorm.py", "AscendRMSNorm", "reset_runtime_state_after_snapshot_restore"),
        ("attention/indexer.py", "AscendSFAIndexerBackend", "rebuild_derived_tensors_after_snapshot_restore"),
        ("attention/indexer_kpool.py", "Glm5NextKPoolIndexerBackend", "rebuild_derived_tensors_after_snapshot_restore"),
        ("attention/mla_v1.py", "AscendMLAImpl", "reset_runtime_state_after_snapshot_restore"),
        ("attention/mla_v1.py", "AscendMLAImpl", "_persist_snapshot_derived_tensors"),
        ("attention/sfa_v1.py", "AscendSFAImpl", "_persist_snapshot_derived_tensors"),
    ],
)
def test_redundant_hooks_are_removed(file, cls, method):
    tree = ast.parse((ROOT / "vllm_ascend" / file).read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    assert method not in {getattr(node, "name", None) for node in owner.body}


@pytest.mark.parametrize("expert_lists", [False, True])
@pytest.mark.parametrize("fused_mc2", [False, True])
def test_w8a8_derived_state_survives_deleted_sources(tmp_path, monkeypatch, expert_lists, fused_mc2):
    namespace, _ = checkpoint_namespace(monkeypatch)
    source = ROOT / "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendW8A8DynamicFusedMoEMethod")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading")
    quant_type = object()
    ns = dict(
        torch=torch,
        QuantType=SimpleNamespace(W8A8=quant_type),
        ACL_FORMAT_FRACTAL_NZ=29,
        torch_npu=SimpleNamespace(npu_format_cast=lambda tensor, fmt: tensor),
        get_ascend_config=lambda: SimpleNamespace(enable_fused_mc2=int(fused_mc2)),
        scale_from_float_to_int64=lambda tensor: tensor.to(torch.int64),
        use_cann_megamoe=lambda config: False,
        get_current_vllm_config=lambda: None,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
    model = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        setattr(model, name, torch.nn.Parameter(torch.arange(24.0).reshape(2, 3, 4), requires_grad=False))
    for name in ("w13_weight_scale", "w2_weight_scale", "w13_weight_offset", "w2_weight_offset"):
        setattr(model, name, torch.nn.Parameter(torch.arange(8.0).reshape(2, 4), requires_grad=False))
    ns["process_weights_after_loading"](
        SimpleNamespace(quant_type=quant_type, use_expert_weight_list=expert_lists),
        model,
    )
    if expert_lists:
        assert not hasattr(model, "w13_weight") and not hasattr(model, "w2_weight_scale")
    assert not any(name.startswith("_snapshot_") for name in model.state_dict())
    inventory.capture_tensor_inventory(model)
    live = inventory.inventory_tensors(model)
    expected = {path: tensor.clone() for path, tensor in live.items()}
    assert expected
    identities = {path: (id(t), t.data_ptr()) for path, t in live.items()}
    path = str(tmp_path / "moe.pth")
    namespace["dump_state_dict"](model, path)
    for _ in range(2):
        with torch.no_grad():
            for tensor in live.values():
                tensor.zero_()
        namespace["restore_state_dict"](model, path, "moe")
        for key, tensor in inventory.inventory_tensors(model).items():
            torch.testing.assert_close(tensor, expected[key])
            assert (id(tensor), tensor.data_ptr()) == identities[key]


def test_no_explicit_persistence_helpers_remain():
    for path in (ROOT / "vllm_ascend").rglob("*.py"):
        assert "tensor_persistence import" not in path.read_text(), path
    assert not (ROOT / "vllm_ascend/snapshot/model_runner_lifecycle/tensor_persistence.py").exists()


def test_expert_map_inventory_preserves_manager_aliases(tmp_path, monkeypatch):
    namespace, _ = checkpoint_namespace(monkeypatch)
    source = ROOT.parent / "vllm/vllm/model_executor/layers/fused_moe/routed_experts.py"
    # Exercise the upstream method now inherited without a snapshot override.
    if not source.exists():
        pytest.skip("requires the paired vLLM source checkout")
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoutedExperts")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "update_expert_map_info")
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
    mapping = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    model = torch.nn.Module()
    model.expert_map_manager = SimpleNamespace(
        local_num_experts=2,
        placement_strategy="linear",
        expert_map=mapping,
        expert_mask=None,
        routing_tables=(mapping.clone(), mapping.clone(), mapping.clone()),
    )
    ns["update_expert_map_info"](model)
    model._ascend_expert_map = mapping.clone()
    inventory.capture_tensor_inventory(model)
    path = str(tmp_path / "maps.pth")
    namespace["dump_state_dict"](model, path)
    expected = mapping.clone()
    addresses = {key: t.data_ptr() for key, t in inventory.inventory_tensors(model).items()}
    for _ in range(2):
        for t in inventory.inventory_tensors(model).values():
            t.zero_()
        namespace["restore_state_dict"](model, path, "maps")
        assert model._expert_map is model.expert_map_manager.expert_map
        for key, t in inventory.inventory_tensors(model).items():
            torch.testing.assert_close(t, expected)
            assert t.data_ptr() == addresses[key]


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_captures_target_and_draft_only_when_enabled(monkeypatch, enabled):
    target, draft = model_with_tensors(), model_with_tensors()
    monkeypatch.setitem(sys.modules, "vllm_ascend.snapshot.model_runner_lifecycle.tensor_inventory", inventory)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle",
        SimpleNamespace(get_drafter_model=lambda runner: draft),
    )
    tree = ast.parse((ROOT / "vllm_ascend/worker/worker.py").read_text())
    method = next(
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name == "load_model"
    )
    namespace = {"set_current_vllm_config": lambda cfg: nullcontext()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "worker.py", "exec"), namespace)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_sleep_mode=False),
            snapshot_config=object() if enabled else None,
            weight_transfer_config=None,
        ),
        model_runner=SimpleNamespace(load_model=Mock(), get_model=lambda: target),
    )
    namespace["load_model"](worker)
    worker.model_runner.load_model.assert_called_once()
    assert hasattr(target, "_snapshot_tensor_paths") is enabled
    assert hasattr(draft, "_snapshot_tensor_paths") is enabled
