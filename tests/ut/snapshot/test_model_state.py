# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.snapshot.model_state import (
    _copy_into_nz_int32_weight,
    restore_state_dict,
)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("scale", torch.zeros(2))


def test_restore_state_dict_copies_parameters_and_buffers(tmp_path) -> None:
    path = tmp_path / "model.pth"
    torch.save(
        {
            "weight": torch.tensor([1.0, 2.0]),
            "scale": torch.tensor([3.0, 4.0]),
        },
        path,
    )
    model = _Model()

    restore_state_dict(model, str(path), "model")

    torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.scale, torch.tensor([3.0, 4.0]))


@patch(
    "vllm_ascend.snapshot.model_state.torch_npu.npu_format_cast",
    side_effect=lambda tensor, _: tensor,
)
def test_copy_into_nz_int32_weight_restores_packed_bytes(mock_format_cast) -> None:
    src = torch.tensor([[0x01020304, -1]], dtype=torch.int32)
    dst = torch.zeros_like(src)

    _copy_into_nz_int32_weight(dst, src)

    torch.testing.assert_close(dst, src)
    mock_format_cast.assert_called_once()


def test_copy_into_nz_int32_weight_rejects_mismatched_state() -> None:
    dst = torch.zeros((1, 2), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="expects int32"):
        _copy_into_nz_int32_weight(dst, torch.zeros((1, 2), dtype=torch.int16))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _copy_into_nz_int32_weight(dst, torch.zeros((2, 2), dtype=torch.int32))
