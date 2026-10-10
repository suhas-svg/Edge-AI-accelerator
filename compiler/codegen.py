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
    if len(out.shape) not in (2, 3):
        raise ValueError(f"tensor {out.name!r} must be rank 2 or 3, got shape {out.shape}")
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


def lower_conv2d(node: Node, specs: dict[str, TensorSpec], base: int) -> list[Command]:
    """Valid-padding stride-1 conv. Encodes m=K, n=OH, k=OW."""
    if node.op != "conv2d":
        raise ValueError(f"lower_conv2d needs a conv2d node, got {node.op!r}")
    if len(node.inputs) != 2:
        raise ValueError(f"conv2d needs 2 inputs, got {len(node.inputs)}")
    for name in (*node.inputs, node.output):
        if name not in specs:
            raise ValueError(f"unknown tensor {name!r}")
    x, w, y = specs[node.inputs[0]], specs[node.inputs[1]], specs[node.output]
    if len(x.shape) != 3:
        raise ValueError(f"conv input must be (C,H,W), got {x.shape}")
    if len(w.shape) != 4:
        raise ValueError(f"conv weight must be (K,C,KH,KW), got {w.shape}")
    if len(y.shape) != 3:
        raise ValueError(f"conv output must be (K,OH,OW), got {y.shape}")
    c, h, wd = x.shape
    k, kc, kh, kw = w.shape
    if kc != c:
        raise ValueError(f"conv channel mismatch weight {kc} vs input {c}")
    if kh != kw:
        raise ValueError(f"conv kernel must be square, got ({kh}, {kw})")
    oh, ow = h - kh + 1, wd - kw + 1
    if y.shape != (k, oh, ow):
        raise ValueError(f"conv output shape {y.shape} != ({k}, {oh}, {ow})")
    for name, dim in (("K", k), ("C", c), ("OH", oh), ("OW", ow)):
        if dim % TILE != 0:
            raise ValueError(f"conv dim {name}={dim} not divisible by {TILE}")
    return [Command(op="CONV2D", address=base,
                    size=_numel(y.shape) * _DTYPE_BYTES[y.dtype], m=k, n=oh, k=ow)]


def lower_max_pool(node: Node, specs: dict[str, TensorSpec], base: int,
                   size: int = 2, stride: int = 2) -> list[Command]:
    if node.op != "max_pool":
        raise ValueError(f"lower_max_pool needs a max_pool node, got {node.op!r}")
    if len(node.inputs) != 1:
        raise ValueError(f"max_pool needs 1 input, got {len(node.inputs)}")
    for name in (*node.inputs, node.output):
        if name not in specs:
            raise ValueError(f"unknown tensor {name!r}")
    if size <= 0 or stride <= 0:
        raise ValueError(f"pool size={size} stride={stride} must be positive")
    x, y = specs[node.inputs[0]], specs[node.output]
    if len(x.shape) != 3 or len(y.shape) != 3:
        raise ValueError(f"pool tensors must be (C,H,W), got {x.shape} → {y.shape}")
    c, h, wd = x.shape
    oh, ow = (h - size) // stride + 1, (wd - size) // stride + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(f"pool window {size} larger than input ({h}, {wd})")
    if y.shape != (c, oh, ow):
        raise ValueError(f"pool output shape {y.shape} != ({c}, {oh}, {ow})")
    return [Command(op="MAX_POOL", address=base,
                    size=_numel(y.shape) * _DTYPE_BYTES[y.dtype], m=size, n=stride)]


def lower_requantize(node: Node, specs: dict[str, TensorSpec], base: int,
                     scale: float, zero_point: int = 0) -> list[Command]:
    """INT32 accumulator → INT8. Scale rides as f32 bits in reserved, zp+128 in m."""
    import struct
    if node.op != "requantize":
        raise ValueError(f"lower_requantize needs a requantize node, got {node.op!r}")
    if len(node.inputs) != 1:
        raise ValueError(f"requantize needs 1 input, got {len(node.inputs)}")
    for name in (*node.inputs, node.output):
        if name not in specs:
            raise ValueError(f"unknown tensor {name!r}")
    if not scale > 0:
        raise ValueError(f"requantize scale must be positive, got {scale}")
    if not -128 <= zero_point <= 127:
        raise ValueError(f"zero_point {zero_point} outside INT8 range")
    acc, out = specs[node.inputs[0]], specs[node.output]
    if acc.dtype != "int32":
        raise ValueError(f"requantize input must be int32, got {acc.dtype}")
    if out.dtype != "int8":
        raise ValueError(f"requantize output must be int8, got {out.dtype}")
    if acc.shape != out.shape:
        raise ValueError(f"requantize shape {acc.shape} != output {out.shape}")
    if _numel(out.shape) % (TILE * TILE) != 0:
        raise ValueError(
            f"tensor {out.name!r} has {_numel(out.shape)} elements, "
            f"not a multiple of {TILE * TILE}"
        )
    (scale_bits,) = struct.unpack("<I", struct.pack("<f", scale))
    return [Command(op="REQUANTIZE", address=base,
                    size=_numel(out.shape) * _DTYPE_BYTES[out.dtype],
                    reserved=scale_bits, m=zero_point + 128)]
