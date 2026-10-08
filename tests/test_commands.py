"""Command encode/decode round trip with fixed bytes."""
from python.edge_npu.commands import Command, encode_commands, decode_commands
import pytest


def test_roundtrip_matmul_sequence():
    cmds = [Command(op="LOAD", address=0x100000, size=4096),
            Command(op="MATMUL", m=64, n=64, k=64),
            Command(op="STORE", address=0x300000, size=1024)]
    assert decode_commands(encode_commands(cmds)) == cmds


def test_bad_magic_raises():
    with pytest.raises(ValueError):
        decode_commands(b"\x00\x00\x01\x00" + b"\x00" * 16)
