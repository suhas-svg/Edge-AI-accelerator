"""Matmul lowering: graph node + shapes → LOAD/MATMUL/STORE commands."""
from __future__ import annotations

from compiler.graph import Node
from python.edge_npu.commands import Command
from python.edge_npu.tensor import TensorSpec

TILE = 8
_DTYPE_BYTES = {"fp32": 4, "int8": 1, "int32": 4}


def lower_matmul(node: Node, specs: dict[str, TensorSpec], base: int) -> list[Command]:
    a = specs[node.inputs[0]]
    w = specs[node.inputs[1]]
    m, ka = a.shape
    kb, n = w.shape
    if ka != kb:
        raise ValueError(f"inner dim mismatch {ka} vs {kb}")
    k = ka
    for name, dim in (("m", m), ("n", n), ("k", k)):
        if dim % TILE != 0:
            raise ValueError(f"dim {name}={dim} not divisible by {TILE}")
    a_bytes = m * k * _DTYPE_BYTES[a.dtype]
    w_bytes = k * n * _DTYPE_BYTES[w.dtype]
    c = specs[node.output]
    c_bytes = m * n * _DTYPE_BYTES[c.dtype]
    return [
        Command(op="LOAD", address=base, size=a_bytes),
        Command(op="LOAD", address=base + a_bytes, size=w_bytes),
        Command(op="MATMUL", m=m, n=n, k=k),
        Command(op="STORE", address=base + a_bytes + w_bytes, size=c_bytes),
    ]
