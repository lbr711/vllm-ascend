from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.snapshot.model_runner_lifecycle.checkpoint import dump_model_runner
from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import (
    get_drafter_model,
    rebuild_model_derived_tensors_after_snapshot_restore,
    reset_modules_runtime_state,
)
from vllm_ascend.snapshot.model_runner_lifecycle.restore import (
    _rebuild_v1_native_resources,
    _reset_target_and_drafter_modules_after_restore,
    _reset_v1_block_tables,
    _reset_v1_input_runtime_state,
    _reset_v1_spec_decode_runtime_state,
    _restore_model_runner_runtime_state,
    restore_model_runner,
)


class _TopKHolder(torch.nn.Module):
    def __init__(self, buffer: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = buffer

    def reset_runtime_state_after_snapshot_restore(self) -> None:
        self.topk_indices_buffer.fill_(-1)


class _BackendSpecificReloadTarget:
    def __init__(self) -> None:
        self.reloaded = False
        self.runtime_reset = False

    def rebuild_derived_tensors_after_snapshot_restore(self, act_dtype: torch.dtype) -> None:
        self.reloaded = True

    def reset_runtime_state_after_snapshot_restore(self) -> None:
        self.runtime_reset = True


class _ImplHolder(torch.nn.Module):
    def __init__(self, impl: object) -> None:
        super().__init__()
        self.impl = impl


class _FailingReloadTarget:
    def rebuild_derived_tensors_after_snapshot_restore(self, act_dtype: torch.dtype) -> None:
        raise RuntimeError("restore failed")


def _make_runner(model, drafter_model):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
            model_config=SimpleNamespace(model="/models/test-model"),
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_config=object()),
        dp_size=2,
        dp_rank=1,
        device=torch.device("cpu"),
        get_model=lambda: model,
        get_draft_model=lambda: drafter_model,
    )


def test_dump_model_runner_dumps_target_and_drafter(tmp_path):
    runner = _make_runner(torch.nn.Module(), torch.nn.Module())

    with (
        patch("vllm_ascend.snapshot.model_runner_lifecycle.checkpoint.get_tp_group") as tp_group,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.checkpoint.dump_state_dict") as dump,
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
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore.get_tp_group") as tp_group,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore._restore_model_checkpoint") as restore_one,
        patch(
            "vllm_ascend.snapshot.model_runner_lifecycle.restore._restore_model_runner_runtime_state"
        ) as restore_runtime,
    ):
        tp_group.return_value.rank_in_group = 3
        restore_model_runner(runner, str(tmp_path))

    assert restore_one.call_count == 2
    assert restore_one.call_args_list[0].args[1] is model
    assert restore_one.call_args_list[0].args[3] == "model"
    assert restore_one.call_args_list[1].args[1] is drafter_model
    assert restore_one.call_args_list[1].args[3] == "drafter"
    restore_runtime.assert_called_once_with(runner)


def test_v1_draft_checkpoint_uses_model_proposer():
    model = torch.nn.Module()

    class ModelProposer:
        def get_model(self):
            return model

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=False),
        drafter=ModelProposer(),
    )
    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.AscendSpecDecodeBaseProposer",
        ModelProposer,
    ):
        assert get_drafter_model(runner) is model


def test_restore_model_runner_runtime_state_runs_all_phases():
    runner = _make_runner(torch.nn.Module(), torch.nn.Module())
    model = runner.get_model()

    with (
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore.reload_cos_and_sin_after_restore") as reload_rope,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore._reset_v2_runner_runtime_state") as reset_runner,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore._reset_v1_runner_runtime_state") as reset_v1,
        patch(
            "vllm_ascend.snapshot.model_runner_lifecycle.restore._reset_target_and_drafter_modules_after_restore"
        ) as reset_modules,
    ):
        _restore_model_runner_runtime_state(runner)

    reload_rope.assert_called_once_with(model)
    reset_runner.assert_called_once_with(runner)
    reset_v1.assert_not_called()
    reset_modules.assert_called_once_with(runner)


def test_restore_model_runner_runtime_state_dispatches_v1():
    model = torch.nn.Module()
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=False),
        get_model=lambda: model,
    )

    with (
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore.reload_cos_and_sin_after_restore"),
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore._reset_v2_runner_runtime_state") as reset_v2,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore._reset_v1_runner_runtime_state") as reset_v1,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore._reset_target_and_drafter_modules_after_restore"),
    ):
        _restore_model_runner_runtime_state(runner)

    reset_v1.assert_called_once_with(runner)
    reset_v2.assert_not_called()


def test_reset_v1_spec_decode_runtime_state():
    runner = SimpleNamespace(
        _draft_token_req_ids=["request"],
        _draft_token_ids=torch.ones((1, 1), dtype=torch.int32),
        _draft_probs=torch.ones((1, 1)),
        _draft_prob_req_ids=["request"],
        prev_num_spec_tokens=1,
        num_spec_tokens=3,
        input_batch=SimpleNamespace(prev_req_id_to_index={"request": 0}),
    )

    _reset_v1_spec_decode_runtime_state(runner)

    assert runner._draft_token_req_ids is None
    assert runner._draft_token_ids is None
    assert runner._draft_probs is None
    assert runner._draft_prob_req_ids is None
    assert runner.prev_num_spec_tokens == 3
    assert runner.input_batch.prev_req_id_to_index is None


def test_reset_v1_input_runtime_state():
    runner = SimpleNamespace(
        use_dcp=True,
        dcp_manager=MagicMock(),
        positions=torch.ones(4),
        _positions_cpu_buf=torch.ones(4),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu_tensor=torch.ones(4),
            num_prompt_tokens_cpu_tensor=torch.ones(4),
        ),
        group_len=SimpleNamespace(gpu=torch.ones(4), cpu=torch.ones(4)),
        group_key_idx=SimpleNamespace(gpu=torch.ones(4), cpu=torch.ones(4)),
        group_key_cache_idx=SimpleNamespace(gpu=torch.ones(4), cpu=torch.ones(4)),
    )

    _reset_v1_input_runtime_state(runner)

    assert torch.count_nonzero(runner.positions) == 0
    assert torch.count_nonzero(runner._positions_cpu_buf) == 0
    assert torch.count_nonzero(runner.input_batch.num_computed_tokens_cpu_tensor) == 0
    assert torch.count_nonzero(runner.input_batch.num_prompt_tokens_cpu_tensor) == 0
    runner.dcp_manager.reset_runtime_state_after_snapshot_restore.assert_called_once_with()
    for staged in (runner.group_len, runner.group_key_idx, runner.group_key_cache_idx):
        assert torch.count_nonzero(staged.gpu) == 0
        assert torch.count_nonzero(staged.cpu) == 0


def test_reset_v1_block_tables():
    buffers = [
        SimpleNamespace(gpu=torch.ones(2), cpu=torch.ones(2)),
        SimpleNamespace(gpu=torch.ones(3), cpu=torch.ones(3)),
    ]
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(block_tables=[SimpleNamespace(block_table=buffer) for buffer in buffers])
        )
    )

    _reset_v1_block_tables(runner)

    for buffer in buffers:
        assert torch.count_nonzero(buffer.gpu) == 0
        assert torch.count_nonzero(buffer.cpu) == 0


def test_rebuild_v1_native_resources_preserves_device_metadata_executor():
    executor = object()
    provider = MagicMock()
    runner = SimpleNamespace(
        device_metadata_executor=executor,
        device_metadata_providers={1: provider},
        reset_encoder_cache=MagicMock(),
        _pending_spec_decode_metadata_copies=[object()],
        kvpp=SimpleNamespace(scheduler=None),
    )

    _rebuild_v1_native_resources(runner)

    assert runner.device_metadata_executor is executor
    provider.enable_device_metadata.assert_not_called()
    runner.reset_encoder_cache.assert_called_once_with()
    assert not runner._pending_spec_decode_metadata_copies


def test_reset_target_and_drafter_modules_after_restore():
    shared_topk = torch.full((4, 8), 23, dtype=torch.int32)
    model = _TopKHolder(shared_topk)
    model.child = _TopKHolder(shared_topk)
    backend = _BackendSpecificReloadTarget()
    model.backend = _ImplHolder(backend)
    drafter = _TopKHolder(shared_topk)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        get_model=lambda: model,
        get_draft_model=lambda: drafter,
    )

    _reset_target_and_drafter_modules_after_restore(runner)

    assert torch.all(shared_topk == -1)
    assert backend.runtime_reset


def test_reload_derived_weights_uses_backend_specific_hook():
    target = _BackendSpecificReloadTarget()

    rebuild_model_derived_tensors_after_snapshot_restore(_ImplHolder(target), torch.bfloat16, "model")

    assert target.reloaded


def test_module_lifecycle_dispatches_module_before_impl():
    calls = []
    impl = MagicMock()
    impl.reset_runtime_state_after_snapshot_restore.side_effect = lambda: calls.append("impl")
    module = _ImplHolder(impl)
    module.reset_runtime_state_after_snapshot_restore = lambda: calls.append("module")

    reset_modules_runtime_state((module,))

    assert calls == ["module", "impl"]


def test_reload_derived_weights_propagates_failure():
    with pytest.raises(RuntimeError, match="restore failed"):
        rebuild_model_derived_tensors_after_snapshot_restore(
            _ImplHolder(_FailingReloadTarget()),
            torch.bfloat16,
            "model",
        )
