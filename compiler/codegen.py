"""Graph lowering: graph nodes + shapes → command stream."""
from __future__ import annotations

import math

from compiler.graph import Node
from python.edge_npu.commands import Command
from python.edge_npu.tensor import TensorSpec

TILE = 8
_DTYPE_BYTES = {"fp32": 4, "int8": 1, "int32": 4}


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def lower_matmul(node: Node, specs: dict[str, TensorSpec], base: int) -> list[Command]:
    if node.op != "matmul":
        raise ValueError(f"lower_matmul needs a matmul node, got {node.op!r}")
    if len(node.inputs) != 2:
        raise ValueError(f"matmul needs 2 inputs, got {len(node.inputs)}")
    for name in (*node.inputs, node.output):
        if name not in specs:
            raise ValueError(f"unknown tensor {name!r}")
    a = specs[node.inputs[0]]
    w = specs[node.inputs[1]]
    c = specs[node.output]
    for t in (a, w, c):
        if len(t.shape) != 2:
            raise ValueError(f"tensor {t.name!r} must be rank 2, got shape {t.shape}")
    m, ka = a.shape
    kb, n = w.shape
    if ka != kb:
        raise ValueError(f"inner dim mismatch {ka} vs {kb}")
    if c.shape != (m, n):
        raise ValueError(f"output shape {c.shape} != matmul result ({m}, {n})")
    k = ka
    for name, dim in (("m", m), ("n", n), ("k", k)):
        if dim % TILE != 0:
            raise ValueError(f"dim {name}={dim} not divisible by {TILE}")
    a_bytes = m * k * _DTYPE_BYTES[a.dtype]
    w_bytes = k * n * _DTYPE_BYTES[w.dtype]
    c_bytes = m * n * _DTYPE_BYTES[c.dtype]
    return [
        Command(op="LOAD", address=base, size=a_bytes),
        Command(op="LOAD", address=base + a_bytes, size=w_bytes),
        Command(op="MATMUL", m=m, n=n, k=k),
        Command(op="STORE", address=base + a_bytes + w_bytes, size=c_bytes),
    ]


def _check_elementwise(node: Node, specs: dict[str, TensorSpec], arity: int) -> TensorSpec:
    if len(node.inputs) != arity:
        raise ValueError(f"{node.op} needs {arity} inputs, got {len(node.inputs)}")
    for name in (*node.inputs, node.output):
        if name not in specs:
            raise ValueError(f"unknown tensor {name!r}")
    out = specs[node.output]
    if len(out.shape) != 2:
        raise ValueError(f"tensor {out.name!r} must be rank 2, got shape {out.shape}")
    if _numel(out.shape) % (TILE * TILE) != 0:
        raise ValueError(
            f"tensor {out.name!r} has {_numel(out.shape)} elements, "
            f"not a multiple of {TILE * TILE}"
        )
    return out


def lower_relu(node: Node, specs: dict[str, TensorSpec], base: int) -> list[Command]:
    if node.op != "relu":
        raise ValueError(f"lower_relu needs a relu node, got {node.op!r}")
    out = _check_elementwise(node, specs, 1)
    if specs[node.inputs[0]].shape != out.shape:
        raise ValueError(
            f"relu input shape {specs[node.inputs[0]].shape} != output {out.shape}"
        )
    return [Command(op="RELU", address=base, size=_numel(out.shape) * _DTYPE_BYTES[out.dtype])]


def lower_bias_add(node: Node, specs: dict[str, TensorSpec], base: int) -> list[Command]:
    if node.op != "bias_add":
        raise ValueError(f"lower_bias_add needs a bias_add node, got {node.op!r}")
    out = _check_elementwise(node, specs, 2)
    data, bias = specs[node.inputs[0]], specs[node.inputs[1]]
    if data.shape != out.shape:
        raise ValueError(f"bias_add input shape {data.shape} != output {out.shape}")
    if bias.shape != (out.shape[-1],) and bias.shape != out.shape:
        raise ValueError(
            f"bias shape {bias.shape} broadcasts neither to last dim {out.shape[-1]} "
            f"nor to {out.shape}"
        )
    return [Command(op="BIAS_ADD", address=base, size=_numel(out.shape) * _DTYPE_BYTES[out.dtype])]
