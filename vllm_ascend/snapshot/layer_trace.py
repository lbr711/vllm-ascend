# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-side transformer-layer summaries for snapshot debugging."""

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import logger

_TENSOR_SLOTS = ("input_hidden", "input_residual", "output_hidden", "output_residual")
_SAMPLE_SIZE = 64
_SUMMARY_SIZE = 11


def _floating_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield only floating tensors from ordinary layer inputs or outputs."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _floating_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floating_tensors(item)


class SnapshotLayerTrace:
    """Collect compact summaries without synchronizing after every layer.

    Hooks write uniformly sampled statistics into one fixed device buffer. The
    buffer is copied to the host only after the complete model forward, so the
    trace cannot insert per-layer D2H synchronization into model execution.
    """

    def __init__(self, model: nn.Module) -> None:
        layers = []
        for name, module in model.named_modules():
            name_parts = name.rsplit(".", 2)
            if (
                len(name_parts) >= 2
                and name_parts[-2] == "layers"
                and name_parts[-1].isdigit()
                and hasattr(module, "layer_idx")
            ):
                layers.append((int(name_parts[-1]), name, module))
        layers.sort(key=lambda item: item[0])

        self._layers = [(layer_idx, name) for layer_idx, name, _ in layers]
        self._summary: torch.Tensor | None = None
        self._handles = []
        for slot, (_, _, module) in enumerate(layers):
            self._handles.append(module.register_forward_hook(self._make_hook(slot), with_kwargs=True))

    @property
    def available(self) -> bool:
        return bool(self._layers)

    def begin(self, device: torch.device) -> None:
        expected_shape = (len(self._layers), len(_TENSOR_SLOTS), _SUMMARY_SIZE)
        if self._summary is None:
            self._summary = torch.zeros(expected_shape, dtype=torch.float32, device=device)
        else:
            self._summary.zero_()

    def _make_hook(self, layer_slot: int):
        def hook(
            module: nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            if self._summary is None:
                return
            input_tensors = list(_floating_tensors((inputs, kwargs)))[:2]
            output_tensors = list(_floating_tensors(output))[:2]
            for tensor_slot, tensor in enumerate(input_tensors):
                self._record(layer_slot, tensor_slot, tensor)
            for tensor_slot, tensor in enumerate(output_tensors, start=2):
                self._record(layer_slot, tensor_slot, tensor)

        return hook

    def _record(self, layer_slot: int, tensor_slot: int, tensor: torch.Tensor) -> None:
        assert self._summary is not None
        flat = tensor.detach().reshape(-1)
        num_samples = min(flat.numel(), _SAMPLE_SIZE)
        if num_samples == 0:
            return

        indices = torch.linspace(
            0,
            flat.numel() - 1,
            num_samples,
            dtype=torch.float32,
            device=flat.device,
        ).to(torch.int64)
        samples = torch.index_select(flat, 0, indices).to(torch.float32)
        row = self._summary[layer_slot, tensor_slot]
        row[0] = flat.numel()
        row[1] = num_samples
        row[2] = samples.sum()
        row[3] = samples.abs().sum()
        row[4] = samples.square().sum()
        row[5] = samples.amin()
        row[6] = samples.amax()
        row[7] = samples[0]
        row[8] = samples[num_samples // 3]
        row[9] = samples[(2 * num_samples) // 3]
        row[10] = samples[-1]

    def log(self, phase: str, trace_id: int, request_ids: tuple[str, ...]) -> None:
        assert self._summary is not None
        summary = self._summary.cpu()
        self._summary = None
        traced_layers = 0
        for (layer_idx, layer_name), layer_rows in zip(self._layers, summary):
            values = {}
            for slot_name, row in zip(_TENSOR_SLOTS, layer_rows):
                if int(row[1]) == 0:
                    values[slot_name] = None
                else:
                    values[slot_name] = tuple(float(value) for value in row.tolist())
            if all(value is None for value in values.values()):
                continue
            traced_layers += 1
            logger.info(
                "[snapshot][trace][layer] phase=%s trace_id=%d requests=%s layer=%d name=%s summaries=%s",
                phase,
                trace_id,
                request_ids,
                layer_idx,
                layer_name,
                values,
            )
        logger.info(
            "[snapshot][trace][layer] phase=%s trace_id=%d requests=%s completed layers=%d",
            phase,
            trace_id,
            request_ids,
            traced_layers,
        )
