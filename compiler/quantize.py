"""INT8 quantization pass: FP32 frontend output → scaled INT8 for lowering.

Symmetric per-tensor quantization: scale = max_abs / 127, zero_point = 0.
"""
from __future__ import annotations

import numpy as np

from compiler.graph import Graph
from python.edge_npu.reference import quantize_int8
from python.edge_npu.tensor import TensorSpec


def choose_scale(arr: np.ndarray) -> float:
    peak = float(np.max(np.abs(arr.astype(np.float64))))
    if peak == 0.0:
        return 1.0
    return peak / 127.0


def quantize_weights(weights: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for name, arr in weights.items():
        if arr.dtype.kind != "f":
            raise ValueError(f"weight {name!r} must be float, got {arr.dtype}")
        scale = choose_scale(arr)
        quantized[name] = quantize_int8(arr.astype(np.float32), scale)
        scales[name] = scale
    return quantized, scales


def quantize_activations(samples: dict[str, list[np.ndarray]]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for name, runs in samples.items():
        if not runs:
            raise ValueError(f"activation {name!r} has no calibration samples")
        peak = max(float(np.max(np.abs(r.astype(np.float64)))) for r in runs)
        scales[name] = peak / 127.0 if peak != 0.0 else 1.0
    return scales


def quantize_specs(graph: Graph, specs: dict[str, TensorSpec],
                   scales: dict[str, float]) -> dict[str, TensorSpec]:
    """INT8 specs for lowering; matmul/conv outputs become INT32 accumulators."""
    accumulators: set[str] = set()
    for node in graph.nodes:
        if node.op in ("matmul", "conv2d"):
            accumulators.add(node.output)
    out: dict[str, TensorSpec] = {}
    for name, spec in specs.items():
        if name in accumulators:
            out[name] = TensorSpec(name=name, dtype="int32", shape=spec.shape)
        else:
            if name not in scales:
                raise ValueError(f"tensor {name!r} has no quantization scale")
            if scales[name] <= 0:
                raise ValueError(f"tensor {name!r} scale must be positive")
            out[name] = TensorSpec(name=name, dtype="int8", shape=spec.shape,
                                   scale=scales[name])
    return out
