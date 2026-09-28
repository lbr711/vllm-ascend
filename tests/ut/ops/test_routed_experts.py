# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, EplbExpertTensorList
from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import restore_state_dict


@pytest.mark.parametrize("snapshot_enabled", [False, True])
def test_snapshot_expert_maps_preserve_manager_aliases(snapshot_enabled):
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    torch.nn.Module.__init__(layer)
    mapping = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    layer.expert_map_manager = SimpleNamespace(
        local_num_experts=2,
        placement_strategy="linear",
        expert_map=mapping,
        expert_mask=None,
        routing_tables=(mapping.clone(), mapping.clone(), mapping.clone()),
    )
    with patch(
        "vllm_ascend.ops.fused_moe.routed_experts.get_current_vllm_config",
        return_value=SimpleNamespace(snapshot_config=object() if snapshot_enabled else None),
    ):
        layer.update_expert_map_info()
    assert layer._expert_map is mapping
    assert ("_expert_map" in layer.state_dict()) == snapshot_enabled
    assert ("expert_global_to_physical" in layer.state_dict()) == snapshot_enabled


def _routed_experts(weight_views):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace(
        get_eplb_weight_views=lambda layer: weight_views,
    )
    return routed_experts


def test_get_expert_weights_flattens_layout_aware_views():
    weights = [torch.randn(2, 3, 4), torch.randn(2, 5)]

    views = list(_routed_experts(weights).get_expert_weights())

    assert [view.shape for view in views] == [torch.Size([2, 12]), torch.Size([2, 5])]
    assert views[0].untyped_storage().data_ptr() == weights[0].untyped_storage().data_ptr()


def test_get_expert_weights_preserves_independent_expert_tensors():
    expert_tensors = [torch.randn(3, 4), torch.randn(3, 4)]

    views = list(_routed_experts([expert_tensors]).get_expert_weights())

    assert len(views) == 1
    assert isinstance(views[0], EplbExpertTensorList)
    assert views[0].shape == torch.Size([2, 3, 4])
    assert all(actual is expected for actual, expected in zip(views[0], expert_tensors))

    buffer = torch.empty_like(views[0])
    assert isinstance(buffer, EplbExpertTensorList)
    assert buffer.shape == views[0].shape
    assert all(tensor.storage_offset() == 0 for tensor in buffer)


def test_get_expert_weights_rejects_unsupported_quantization():
    with pytest.raises(NotImplementedError, match="weight views are not defined"):
        list(_routed_experts([]).get_expert_weights())


def test_get_expert_weights_rejects_missing_weight_view_contract():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="must implement get_eplb_weight_views"):
        list(routed_experts.get_expert_weights())


def test_get_expert_weights_rejects_non_expert_first_dimension():
    with pytest.raises(ValueError, match="first dimension"):
        list(_routed_experts([torch.randn(3, 4)]).get_expert_weights())


def test_get_expert_weights_rejects_wrong_expert_tensor_list_length():
    with pytest.raises(ValueError, match="must contain local_num_experts"):
        list(_routed_experts([[torch.randn(3, 4)]]).get_expert_weights())


def test_get_expert_weights_rejects_non_contiguous_view():
    with pytest.raises(ValueError, match="flattenable without a copy"):
        list(_routed_experts([torch.randn(2, 3, 4).transpose(1, 2)]).get_expert_weights())


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_ascend_expert_map_follows_model_runner(use_v2_model_runner):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    torch.nn.Module.__init__(routed_experts)
    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    upstream_map = torch.tensor([0, 1], dtype=torch.int32)
    object.__setattr__(routed_experts, "_use_v2_model_runner", use_v2_model_runner)
    # Both v0.28.0 and main read quant_method in RoutedExperts.expert_map.
    routed_experts.quant_method = SimpleNamespace(moe_kernel=None)
    routed_experts.ascend_expert_map = legacy_map
    object.__setattr__(routed_experts, "_expert_map", upstream_map)
    object.__setattr__(routed_experts, "rocm_aiter_fmoe_enabled", False)

    expected = upstream_map if use_v2_model_runner else legacy_map
    assert routed_experts.ascend_expert_map is expected


def test_update_expert_map_preserves_upstream_and_legacy_contracts(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    torch.nn.Module.__init__(routed_experts)
    parent_update_calls = []

    def parent_update(instance):
        parent_update_calls.append(instance)

    monkeypatch.setattr(type(routed_experts).__mro__[1], "update_expert_map", parent_update)
    expert_map_manager = SimpleNamespace(_expert_map=None)
    object.__setattr__(routed_experts, "expert_map_manager", expert_map_manager)

    routed_experts.update_expert_map()

    assert parent_update_calls == [routed_experts]

    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    routed_experts.update_expert_map(legacy_map)

    assert routed_experts.ascend_expert_map is legacy_map
    assert expert_map_manager._expert_map is legacy_map


@pytest.mark.parametrize("snapshot_enabled", [False, True])
@pytest.mark.parametrize("has_map", [False, True])
def test_v1_execution_map_checkpoint_round_trip(snapshot_enabled, has_map, tmp_path):
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    torch.nn.Module.__init__(layer)
    layer._use_v2_model_runner = False
    layer.mix_placement = False
    layer.moe_config = SimpleNamespace(num_experts=4, num_logical_experts=4, num_local_experts=2, ep_size=2)
    mapping = torch.tensor([-1, -1, 0, 1], dtype=torch.int32) if has_map else None
    expected = mapping.clone() if has_map else None
    layer.expert_map_manager = SimpleNamespace(_expert_map=None)
    config = SimpleNamespace(
        snapshot_config=object() if snapshot_enabled else None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    eplb = SimpleNamespace(
        num_redundant_experts=0, dynamic_eplb=False, eplb_policy_type=0, expert_heat_collection_interval=1
    )
    prefix = "vllm_ascend.ops.fused_moe.routed_experts"
    with patch(prefix + ".get_current_vllm_config", return_value=config), patch(
        prefix + ".get_ascend_config", return_value=SimpleNamespace(eplb_config=eplb)
    ), patch(prefix + ".make_eplb_placement_config", return_value=eplb), patch(
        prefix + ".init_eplb_config", return_value=(None, mapping, None, 0)
    ), patch(prefix + ".use_multistage_eplb_load", return_value=False), patch(
        prefix + ".VllmEplbAdaptor.register_layer"
    ), patch.object(torch.Tensor, "npu", lambda tensor: tensor, create=True):
        layer.init_eplb(0)
    assert layer.ascend_expert_map is mapping
    assert ("_ascend_expert_map" in layer._buffers) == snapshot_enabled
    assert ("_ascend_expert_map" in layer.state_dict()) == (snapshot_enabled and has_map)
    if snapshot_enabled and has_map:
        path = tmp_path / "map.pth"
        torch.save(layer.state_dict(), path)
        address = mapping.data_ptr()
        for _ in range(2):
            mapping.zero_()
            restore_state_dict(layer, str(path), "model")
            assert layer.ascend_expert_map is mapping
            assert mapping.data_ptr() == address
            torch.testing.assert_close(mapping, expected)
        replacement = expected.clone()
        layer.update_expert_map(replacement)
        assert layer.ascend_expert_map is replacement
        assert layer._buffers["_ascend_expert_map"] is replacement
        assert "_ascend_expert_map" not in layer.__dict__
        assert layer.expert_map_manager._expert_map is replacement


def test_v2_execution_map_checkpoint_preserves_manager_aliases(tmp_path):
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    torch.nn.Module.__init__(layer)
    layer._use_v2_model_runner = True
    layer.quant_method = SimpleNamespace(moe_kernel=None)
    mapping = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    tables = tuple(torch.arange(4, dtype=torch.int32) for _ in range(3))
    layer.expert_map_manager = SimpleNamespace(
        local_num_experts=2, placement_strategy="linear", expert_map=mapping, expert_mask=None, routing_tables=tables
    )
    with patch(
        "vllm_ascend.ops.fused_moe.routed_experts.get_current_vllm_config",
        return_value=SimpleNamespace(snapshot_config=object()),
    ):
        layer.update_expert_map_info()
    path = tmp_path / "v2_map.pth"
    torch.save(layer.state_dict(), path)
    for _ in range(2):
        mapping.zero_()
        for table in tables:
            table.zero_()
        restore_state_dict(layer, str(path), "model")
        assert layer.ascend_expert_map is layer.expert_map_manager.expert_map
        torch.testing.assert_close(layer.ascend_expert_map, torch.tensor([0, 1, -1, -1], dtype=torch.int32))
        for name, table in zip(
            ("expert_global_to_physical", "expert_physical_to_global", "expert_local_to_global"), tables
        ):
            assert getattr(layer, name) is table
            torch.testing.assert_close(table, torch.arange(4, dtype=torch.int32))
