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

def test_roundtrip_elementwise_v02():
    cmds = [Command(op="RELU", address=0x3000, size=16384),
            Command(op="BIAS_ADD", address=0x3000, size=16384)]
    assert decode_commands(encode_commands(cmds)) == cmds


def test_roundtrip_conv_pool_v03():
    cmds = [Command(op="CONV2D", address=0x3000, size=2048, m=8, n=8, k=8),
            Command(op="MAX_POOL", address=0x4000, size=512, m=2, n=2)]
    assert decode_commands(encode_commands(cmds)) == cmds


def test_roundtrip_requantize_v04():
    import struct
    scale_bits = struct.unpack("<I", struct.pack("<f", 0.05))[0]
    cmds = [Command(op="REQUANTIZE", address=0x5000, size=64,
                    reserved=scale_bits, m=128)]
    assert decode_commands(encode_commands(cmds)) == cmds
