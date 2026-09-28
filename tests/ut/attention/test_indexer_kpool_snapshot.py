from unittest.mock import patch

import torch

from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolMetadataBuilder, Glm5NextKPoolIndexerBackend


def test_kpool_builder_discards_per_step_metadata_before_recapture():
    builder = AscendIndexerKPoolMetadataBuilder.__new__(AscendIndexerKPoolMetadataBuilder)
    builder._metadata_buffers = {123: (torch.ones(4),)}
    builder.reset_runtime_state_after_snapshot_restore()
    builder.reset_runtime_state_after_snapshot_restore()
    assert builder._metadata_buffers == {}


def test_kpool_indexer_rebuilds_derived_weights_after_restore():
    backend = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)

    with patch.object(
        Glm5NextKPoolIndexerBackend,
        "process_weights_after_loading",
    ) as process_weights:
        backend.rebuild_derived_tensors_after_snapshot_restore(torch.bfloat16)

    process_weights.assert_called_once_with()


def test_kpool_indexer_clears_topk_buffer_after_restore():
    backend = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)
    backend.topk_indices_buffer = torch.zeros((2, 4), dtype=torch.int32)

    backend.reset_runtime_state_after_snapshot_restore()

    assert torch.all(backend.topk_indices_buffer == -1)
