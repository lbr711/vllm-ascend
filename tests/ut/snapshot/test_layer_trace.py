# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from vllm_ascend.snapshot.layer_trace import SnapshotLayerTrace


class _Layer(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self, positions: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
        return hidden + 1, residual + 2


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(0), _Layer(1)])


def test_layer_trace_keeps_input_and_output_slots_separate() -> None:
    model = _Model()
    trace = SnapshotLayerTrace(model)
    trace.begin(torch.device("cpu"))

    positions = torch.arange(4, dtype=torch.int64)
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    hidden, residual = model.layers[0](positions, hidden, None)
    model.layers[1](positions, hidden, residual)

    assert trace._summary is not None
    layer_zero = trace._summary[0]
    assert layer_zero[0, 1] == 8
    assert layer_zero[1, 1] == 0
    assert layer_zero[2, 1] == 8
    assert layer_zero[3, 1] == 8
    assert layer_zero[0, 2] == 28
    assert layer_zero[2, 2] == 36
    assert layer_zero[3, 2] == 44
