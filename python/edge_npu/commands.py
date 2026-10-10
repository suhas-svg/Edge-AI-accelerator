"""Hardware command interface (spec sections 7, 9). Binary encoding v0.2."""
from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Literal

OpCode = Literal["LOAD", "MATMUL", "STORE", "RELU", "BIAS_ADD"]
OP_IDS: dict[str, int] = {"LOAD": 0x01, "MATMUL": 0x02, "STORE": 0x03,
                          "RELU": 0x04, "BIAS_ADD": 0x05}
ID_OPS: dict[int, str] = {v: k for k, v in OP_IDS.items()}
_HEADER = struct.Struct("<HI")
_RECORD = struct.Struct("<BIIIHHH")


@dataclass(frozen=True)
class Command:
    op: OpCode
    address: int = 0
    size: int = 0
    m: int = 0
    n: int = 0
    k: int = 0


def encode_commands(cmds: list[Command]) -> bytes:
    out = bytearray()
    out += _HEADER.pack(0x454E, len(cmds))  # magic 'NE', count
    for c in cmds:
        out += _RECORD.pack(OP_IDS[c.op], c.address, c.size, 0, c.m, c.n, c.k)
    return bytes(out)


def decode_commands(buf: bytes) -> list[Command]:
    if len(buf) < _HEADER.size:
        raise ValueError("buffer too short for header")
    magic, count = _HEADER.unpack_from(buf, 0)
    if magic != 0x454E:
        raise ValueError(f"bad magic {magic:#x}")
    cmds: list[Command] = []
    off = _HEADER.size
    for _ in range(count):
        op_id, addr, size, _, m, n, k = _RECORD.unpack_from(buf, off)
        if op_id not in ID_OPS:
            raise ValueError(f"unknown opcode {op_id:#x}")
        cmds.append(Command(op=ID_OPS[op_id], address=addr, size=size, m=m, n=n, k=k))  # type: ignore[typeddict-item]
        off += _RECORD.size
    return cmds
