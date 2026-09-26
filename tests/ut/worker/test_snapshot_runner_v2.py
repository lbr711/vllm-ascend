# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.snapshot.model_runtime.restore import (
    _reset_block_tables,
    _reset_input_buffers,
    _reset_request_state,
    _reset_speculator,
    get_drafter_model,
    reset_graph_managers,
)


def _staged():
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    buffer = StagedWriteTensor.__new__(StagedWriteTensor)
    buffer.gpu = torch.ones(4)
    buffer._staged_write_indices = [1]
    buffer._staged_write_starts = [0]
    buffer._staged_write_contents = [7]
    buffer._staged_write_cu_lens = [1]
    return buffer


def _uva():
    return SimpleNamespace(cpu=torch.ones(4), copy_to_uva=MagicMock())


def test_v2_draft_checkpoint_uses_runner_interface():
    model = torch.nn.Linear(2, 2)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        get_draft_model=lambda: model,
    )
    assert get_drafter_model(runner) is model


def test_v2_input_buffers_reset_preserves_storage():
    names = ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens", "dcp_local_seq_lens")
    buffers = SimpleNamespace(**{name: torch.ones(4) for name in names})
    pointers = {name: getattr(buffers, name).data_ptr() for name in names}
    _reset_input_buffers(buffers)
    for name in names:
        assert torch.count_nonzero(getattr(buffers, name)) == 0
        assert getattr(buffers, name).data_ptr() == pointers[name]


def test_v2_block_tables_reset_rebuilds_pointer_metadata():
    table = _staged()
    tables = SimpleNamespace(
        block_tables=[table],
        num_blocks=_uva(),
        input_block_tables=[torch.ones(4)],
        slot_mappings=torch.ones(4),
        init_block_table_layout_tensors=MagicMock(),
    )
    _reset_block_tables(tables)
    assert not table._staged_write_indices
    assert not table._staged_write_contents
    assert torch.count_nonzero(table.gpu) == 0
    assert torch.count_nonzero(tables.input_block_tables[0]) == 0
    assert torch.all(tables.slot_mappings == -1)
    tables.num_blocks.copy_to_uva.assert_called_once()
    tables.init_block_table_layout_tensors.assert_called_once()


def test_v2_request_state_reset_and_idle_contract():
    state = SimpleNamespace(
        num_reqs=1,
        all_token_ids=_staged(),
        total_len=_staged(),
        num_computed_tokens=_staged(),
        prompt_len=_uva(),
        prefill_len=_uva(),
        num_computed_prefill_tokens=np.ones(4),
        num_computed_tokens_np=np.ones(4),
        max_seq_len=np.ones(4),
        last_sampled_tokens=torch.ones(4),
        draft_tokens=torch.ones(4),
        next_prefill_tokens=torch.ones(4),
    )
    with pytest.raises(RuntimeError, match="empty Model Runner V2 request state"):
        _reset_request_state(state)
    state.num_reqs = 0
    _reset_request_state(state)
    _reset_request_state(state)
    assert not np.any(state.num_computed_tokens_np)
    assert torch.count_nonzero(state.next_prefill_tokens) == 0
    assert not state.num_computed_tokens._staged_write_contents


def test_v2_graph_managers_recreated_without_loading_or_compiling_model():
    runner = SimpleNamespace(
        vllm_config=object(),
        device=torch.device("cpu"),
        decode_query_len=4,
        cudagraph_manager=SimpleNamespace(cudagraph_mode=1, lora_capture_cases=[0], varlen_decode=False),
        speculator=SimpleNamespace(init_cudagraph_manager=MagicMock()),
    )
    with patch("vllm_ascend.worker.v2.aclgraph_utils.ModelAclGraphManager") as manager:
        reset_graph_managers(runner)
    assert runner.cudagraph_manager is manager.return_value
    runner.speculator.init_cudagraph_manager.assert_called_once_with(1)


def test_v2_mtp_resets_step_indices_and_draft_inputs():
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator

    class Speculator(AutoRegressiveSpeculator):
        def load_draft_model(self, *args):
            raise AssertionError("Restore must not reload or compile the draft model")

    speculator = Speculator.__new__(Speculator)
    names = (
        "input_ids",
        "positions",
        "is_padding",
        "query_start_loc",
        "seq_lens",
        "dcp_local_seq_lens",
        "seq_lens_cpu",
    )
    speculator.input_buffers = SimpleNamespace(**{name: torch.ones(4) for name in names})
    for name in (
        "idx_mapping",
        "temperature",
        "seeds",
        "draft_tokens",
        "hidden_states",
        "current_draft_step",
        "last_token_indices",
        "sample_src_positions",
    ):
        setattr(speculator, name, torch.ones(4))
    speculator.draft_logits = None
    speculator.inputs_embeds = None
    speculator.input_batch = object()
    _reset_speculator(speculator)
    assert speculator.input_batch is None
    assert torch.count_nonzero(speculator.current_draft_step) == 0
    assert torch.count_nonzero(speculator.last_token_indices) == 0
    assert torch.count_nonzero(speculator.input_buffers.positions) == 0
