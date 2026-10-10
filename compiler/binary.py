"""Versioned model.bin writer and reader (docs/model-format.md)."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from compiler.graph import Graph, Node
from python.edge_npu.commands import Command, decode_commands, encode_commands

MODEL_VERSION: int = 1
_MAGIC = 0x454D
_HEADER = struct.Struct("<HHII")
_DTYPE_IDS = {"fp32": 0, "int8": 1, "int32": 2}
_ID_DTYPES = {v: k for k, v in _DTYPE_IDS.items()}
_NP_DTYPES = {"fp32": np.float32, "int8": np.int8, "int32": np.int32}
_DTYPE_RANGES = {"int8": (-128, 127), "int32": (-(2 ** 31), 2 ** 31 - 1)}


@dataclass
class ModelPack:
    version: int
    graph: Graph
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    cmds: list[Command] = field(default_factory=list)


def _np_dtype_name(arr: np.ndarray) -> str:
    if arr.dtype == np.int8:
        return "int8"
    if arr.dtype == np.int32:
        return "int32"
    if arr.dtype == np.float32:
        return "fp32"
    raise ValueError(f"weight dtype {arr.dtype} not packable (need int8/int32/fp32)")


def write_model(path: str, graph: Graph, weights: dict[str, np.ndarray],
                cmds: list[Command]) -> None:
    names = sorted(weights)
    blobs: dict[str, bytes] = {}
    for name in names:
        arr = np.ascontiguousarray(weights[name])
        dtype = _np_dtype_name(arr)
        if dtype in _DTYPE_RANGES:
            lo, hi = _DTYPE_RANGES[dtype]
            if arr.min() < lo or arr.max() > hi:
                raise ValueError(f"weight {name!r} out of {dtype} range")
        blobs[name] = arr.tobytes()
    cmd_bytes = encode_commands(cmds)
    out = bytearray()
    out += _HEADER.pack(_MAGIC, MODEL_VERSION, len(names), len(cmd_bytes))
    for name in names:
        arr = weights[name]
        dtype = _np_dtype_name(np.asarray(arr))
        nb = name.encode("utf-8")
        if len(nb) > 255:
            raise ValueError(f"tensor name {name!r} too long")
        out += struct.pack("<B", len(nb)) + nb
        out += struct.pack("<BB", _DTYPE_IDS[dtype], len(arr.shape))
        for d in arr.shape:
            out += struct.pack("<I", d)
        out += struct.pack("<f", 1.0) + struct.pack("<b", 0)
    for name in names:
        out += blobs[name]
    out += struct.pack("<I", len(graph.nodes))
    for n in graph.nodes:
        opb = n.op.encode("utf-8")
        out += struct.pack("<B", len(opb)) + opb
        out += struct.pack("<B", len(n.inputs))
        for inp in n.inputs:
            ib = inp.encode("utf-8")
            out += struct.pack("<B", len(ib)) + ib
        ob = n.output.encode("utf-8")
        out += struct.pack("<B", len(ob)) + ob
    out += cmd_bytes
    with open(path, "wb") as f:
        f.write(bytes(out))


def _need(buf: bytes, off: int, n: int, section: str) -> None:
    if len(buf) < off + n:
        raise ValueError(f"model.bin truncated in {section}")


def read_model(path: str) -> ModelPack:
    with open(path, "rb") as f:
        buf = f.read()
    _need(buf, 0, _HEADER.size, "header")
    magic, version, tensor_count, cmd_len = _HEADER.unpack_from(buf, 0)
    if magic != _MAGIC:
        raise ValueError(f"bad model magic {magic:#x}")
    if version != MODEL_VERSION:
        raise ValueError(f"unknown model version {version}")
    off = _HEADER.size
    specs: list[tuple[str, str, tuple[int, ...]]] = []
    for _ in range(tensor_count):
        _need(buf, off, 1, "tensor table")
        (name_len,) = struct.unpack_from("<B", buf, off)
        off += 1
        _need(buf, off, name_len + 2, "tensor table")
        name = buf[off:off + name_len].decode("utf-8")
        dtype_id, rank = struct.unpack_from("<BB", buf, off + name_len)
        off += name_len + 2
        if dtype_id not in _ID_DTYPES:
            raise ValueError(f"unknown dtype id {dtype_id}")
        _need(buf, off, 4 * rank + 5, "tensor table")
        dims = struct.unpack_from("<" + "I" * rank, buf, off) if rank else ()
        off += 4 * rank
        off += 4 + 1  # scale f32 + zero point i8
        specs.append((name, _ID_DTYPES[dtype_id], tuple(dims)))
    weights: dict[str, np.ndarray] = {}
    for name, dtype, dims in specs:
        n = int(np.prod(dims)) if dims else 0
        size = n * np.dtype(_NP_DTYPES[dtype]).itemsize
        _need(buf, off, size, f"weights for {name!r}")
        weights[name] = np.frombuffer(buf[off:off + size], dtype=_NP_DTYPES[dtype]).reshape(dims).copy()
        off += size
    _need(buf, off, 4, "node table")
    (node_count,) = struct.unpack_from("<I", buf, off)
    off += 4
    nodes: list[Node] = []
    for _ in range(node_count):
        _need(buf, off, 1, "node table")
        (op_len,) = struct.unpack_from("<B", buf, off)
        off += 1
        _need(buf, off, op_len + 1, "node table")
        op = buf[off:off + op_len].decode("utf-8")
        off += op_len
        (in_count,) = struct.unpack_from("<B", buf, off)
        off += 1
        inputs: list[str] = []
        for _ in range(in_count):
            _need(buf, off, 1, "node table")
            (il,) = struct.unpack_from("<B", buf, off)
            off += 1
            _need(buf, off, il, "node table")
            inputs.append(buf[off:off + il].decode("utf-8"))
            off += il
        _need(buf, off, 1, "node table")
        (ol,) = struct.unpack_from("<B", buf, off)
        off += 1
        _need(buf, off, ol, "node table")
        output = buf[off:off + ol].decode("utf-8")
        off += ol
        nodes.append(Node(op=op, inputs=tuple(inputs), output=output))
    _need(buf, off, cmd_len, "command stream")
    try:
        cmds = decode_commands(buf[off:off + cmd_len])
    except struct.error as e:
        raise ValueError(f"model.bin truncated in command stream: {e}")
    return ModelPack(version=version, graph=Graph(nodes=tuple(nodes)), weights=weights, cmds=cmds)
