from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.snapshot.model_runtime.restore import (
    _reset_block_table_runtime_state,
    _reset_model_module_runtime_state,
    _reset_runner_input_runtime_state,
    _restore_model_runner_runtime_state,
    dump_model_runner,
    restore_model_runner,
)
from vllm_ascend.snapshot.model_runtime.tensor_lifecycle import restore_derived_tensor_state


class _TopKHolder(torch.nn.Module):
    def __init__(self, buffer: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = buffer

    def reset_snapshot_runtime_state(self) -> None:
        self.topk_indices_buffer.fill_(-1)


class _BackendSpecificReloadTarget:
    def __init__(self) -> None:
        self.reloaded = False

    def restore_snapshot_derived_state(self, act_dtype: torch.dtype) -> None:
        self.reloaded = True


class _ImplHolder(torch.nn.Module):
    def __init__(self, impl: object) -> None:
        super().__init__()
        self.impl = impl


class _FailingReloadTarget:
    def restore_snapshot_derived_state(self, act_dtype: torch.dtype) -> None:
        raise RuntimeError("restore failed")


def _make_runner(model, drafter_model):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
            model_config=SimpleNamespace(model="/models/test-model"),
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_config=object()),
        dp_size=2,
        dp_rank=1,
        device=torch.device("cpu"),
        drafter=SimpleNamespace(model=drafter_model),
        get_model=lambda: model,
    )


def test_dump_model_runner_dumps_target_and_drafter(tmp_path):
    runner = _make_runner(torch.nn.Module(), torch.nn.Module())

    with (
        patch("vllm_ascend.snapshot.model_runtime.restore.get_tp_group") as tp_group,
        patch("vllm_ascend.snapshot.model_runtime.restore.dump_state_dict") as dump,
    ):
        tp_group.return_value.rank_in_group = 3
        dump_model_runner(runner, str(tmp_path))

    assert dump.call_count == 2
    assert str(dump.call_args_list[0].args[1]).endswith("model_ckpt.1tp3.pth")
    assert str(dump.call_args_list[1].args[1]).endswith("model_ckpt_drafter.1tp3.pth")


def test_restore_model_runner_restores_target_and_drafter(tmp_path):
    model = torch.nn.Module()
    drafter_model = torch.nn.Module()
    runner = _make_runner(model, drafter_model)

    with (
        patch("vllm_ascend.snapshot.model_runtime.restore.get_tp_group") as tp_group,
        patch("vllm_ascend.snapshot.model_runtime.restore._restore_model_checkpoint") as restore_one,
        patch("vllm_ascend.snapshot.model_runtime.restore._restore_model_runner_runtime_state") as restore_runtime,
    ):
        tp_group.return_value.rank_in_group = 3
        restore_model_runner(runner, str(tmp_path))

    assert restore_one.call_count == 2
    assert restore_one.call_args_list[0].args[1] is model
    assert restore_one.call_args_list[0].args[3] == "model"
    assert restore_one.call_args_list[1].args[1] is drafter_model
    assert restore_one.call_args_list[1].args[3] == "drafter"
    restore_runtime.assert_called_once_with(runner, model)


def test_restore_model_runner_runtime_state_runs_all_phases():
    runner = _make_runner(torch.nn.Module(), torch.nn.Module())
    model = runner.get_model()

    with (
        patch("vllm_ascend.snapshot.model_runtime.restore.restore_global_tensor_state") as restore_global,
        patch("vllm_ascend.snapshot.model_runtime.restore._reset_spec_decode_runtime_state") as reset_spec,
        patch("vllm_ascend.snapshot.model_runtime.restore._restore_drafter_runtime_state") as restore_drafter,
        patch("vllm_ascend.snapshot.model_runtime.restore._reset_attention_builder_runtime_state") as reset_attention,
        patch("vllm_ascend.snapshot.model_runtime.restore._reset_runner_input_runtime_state") as reset_runner,
        patch("vllm_ascend.snapshot.model_runtime.restore._reset_model_module_runtime_state") as reset_modules,
        patch("vllm_ascend.snapshot.model_runtime.restore._reset_block_table_runtime_state") as reset_block_table,
    ):
        _restore_model_runner_runtime_state(runner, model)

    restore_global.assert_called_once_with(model, runner.model_config.hf_config, runner.device)
    reset_spec.assert_called_once_with(runner)
    restore_drafter.assert_called_once_with(runner)
    reset_attention.assert_called_once_with(runner)
    reset_runner.assert_called_once_with(runner)
    reset_modules.assert_called_once_with(runner)
    reset_block_table.assert_called_once_with(runner)


def test_reset_runner_input_runtime_state():
    runner = SimpleNamespace()
    runner.use_dcp = True
    runner.dcp_manager = MagicMock()
    runner.positions = torch.full((4,), 29, dtype=torch.int64)
    runner._positions_cpu_buf = torch.full((4,), 31, dtype=torch.int64)
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu_tensor=torch.full((4,), 37, dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.full((4,), 41, dtype=torch.int32),
    )
    runner.group_len = SimpleNamespace(
        gpu=torch.full((4,), 3, dtype=torch.int32),
        cpu=torch.full((4,), 5, dtype=torch.int32),
    )
    runner.group_key_idx = SimpleNamespace(
        gpu=torch.full((4,), 7, dtype=torch.int32),
        cpu=torch.full((4,), 11, dtype=torch.int32),
    )
    runner.group_key_cache_idx = SimpleNamespace(
        gpu=torch.full((4,), 13, dtype=torch.int32),
        cpu=torch.full((4,), 17, dtype=torch.int32),
    )

    _reset_runner_input_runtime_state(runner)

    for staged in (
        runner.group_len,
        runner.group_key_idx,
        runner.group_key_cache_idx,
    ):
        assert torch.count_nonzero(staged.gpu) == 0
        assert torch.count_nonzero(staged.cpu) == 0
    assert torch.count_nonzero(runner.positions) == 0
    assert torch.count_nonzero(runner._positions_cpu_buf) == 0
    assert torch.count_nonzero(runner.input_batch.num_computed_tokens_cpu_tensor) == 0
    assert torch.count_nonzero(runner.input_batch.num_prompt_tokens_cpu_tensor) == 0
    runner.dcp_manager.reset_snapshot_runtime_state.assert_called_once_with()


def test_reset_model_module_runtime_state():
    shared_topk = torch.full((4, 8), 23, dtype=torch.int32)
    model = _TopKHolder(shared_topk)
    model.child = _TopKHolder(shared_topk)
    drafter = _TopKHolder(shared_topk)
    runner = SimpleNamespace(
        get_model=lambda: model,
        drafter=SimpleNamespace(model=drafter),
    )

    _reset_model_module_runtime_state(runner)

    assert torch.all(shared_topk == -1)


def test_reload_derived_weights_uses_backend_specific_hook():
    target = _BackendSpecificReloadTarget()

    restore_derived_tensor_state(_ImplHolder(target), torch.bfloat16, "model")

    assert target.reloaded


def test_reload_derived_weights_propagates_failure():
    with pytest.raises(RuntimeError, match="restore failed"):
        restore_derived_tensor_state(
            _ImplHolder(_FailingReloadTarget()),
            torch.bfloat16,
            "model",
        )


def test_reset_block_tables_clears_cpu_and_device_buffers():
    runner = SimpleNamespace()
    buffers = [
        SimpleNamespace(gpu=torch.ones(2), cpu=torch.ones(2)),
        SimpleNamespace(gpu=torch.ones(3), cpu=torch.ones(3)),
    ]
    block_table = SimpleNamespace(block_tables=[SimpleNamespace(block_table=buf) for buf in buffers])
    runner.input_batch = SimpleNamespace(block_table=block_table)

    _reset_block_table_runtime_state(runner)

    for buf in buffers:
        assert torch.count_nonzero(buf.gpu) == 0
        assert torch.count_nonzero(buf.cpu) == 0
