import torch

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder


def test_metadata_builder_reset_restores_cold_start_state():
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    builder.num_decodes = 4
    builder.num_prefills = 3
    builder.num_decode_tokens = 8
    builder.num_prefill_tokens = 12
    builder.num_actual_tokens = 20
    builder.cu_seq_lens_cpu = torch.ones(2)
    builder.context_lens_cpu = torch.ones(2)
    builder.block_table = torch.ones((2, 2), dtype=torch.int32)
    builder.graph_pad_size = 7
    builder.query_lens = torch.ones(2, dtype=torch.int32)
    builder.seq_lens = torch.ones(2, dtype=torch.int32)
    builder.prefill_ratio_to_sas_metadata = {"stale": object()}
    builder.decode_ratio_to_sas_metadata = {"stale": object()}
    builder.common_ratio_to_sas_metadata = {"stale": object()}

    tensor_names = (
        "start_pos_prefill",
        "start_pos_decode",
        "decode_sas_metadata",
        "decode_qli_metadata",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "seqused_q",
        "_zero_i32",
        "slot_mapping",
    )
    for name in tensor_names:
        setattr(builder, name, torch.ones(2, dtype=torch.int32))
    builder.spec_slot_mapping = [torch.ones(2, dtype=torch.int32)]
    builder.spec_sas_metadata = [torch.ones(2, dtype=torch.int32)]

    builder.reset_runtime_state_after_snapshot_restore()

    assert builder.num_decodes == 0
    assert builder.num_prefills == 0
    assert builder.num_decode_tokens == 0
    assert builder.num_prefill_tokens == 0
    assert builder.num_actual_tokens is None
    assert builder.cu_seq_lens_cpu is None
    assert builder.context_lens_cpu is None
    assert builder.block_table is None
    assert builder.graph_pad_size == 0
    assert builder.query_lens is None
    assert builder.seq_lens is None
    assert builder.prefill_ratio_to_sas_metadata == {}
    assert builder.decode_ratio_to_sas_metadata == {}
    assert builder.common_ratio_to_sas_metadata == {}
    assert all(torch.count_nonzero(getattr(builder, name)) == 0 for name in tensor_names)
    assert torch.count_nonzero(builder.spec_slot_mapping[0]) == 0
    assert torch.count_nonzero(builder.spec_sas_metadata[0]) == 0
