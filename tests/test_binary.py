import struct

import numpy as np
import pytest

from compiler.binary import read_model, write_model
from compiler.graph import Graph, Node
from python.edge_npu.commands import Command


def test_roundtrip_single_matmul(tmp_path):
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    weights = {"w": np.ones((8, 8), dtype=np.int8)}
    cmds = [Command(op="LOAD", address=0x1000, size=64),
            Command(op="MATMUL", m=8, n=8, k=8),
            Command(op="STORE", address=0x3000, size=256)]
    path = str(tmp_path / "tiny.bin")
    write_model(path, graph, weights, cmds)
    pack = read_model(path)
    assert pack.version == 1
    assert pack.graph == graph
    assert pack.weights["w"].tolist() == weights["w"].tolist()
    assert pack.cmds == cmds


def test_unknown_version_raises(tmp_path):
    path = str(tmp_path / "bad.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<HHII", 0x454D, 99, 0, 0))
    with pytest.raises(ValueError):
        read_model(path)


def test_truncated_raises(tmp_path):
    path = str(tmp_path / "short.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<HHII", 0x454D, 1, 1, 0))
    with pytest.raises(ValueError):
        read_model(path)


def test_out_of_range_weight_raises(tmp_path):
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    weights = {"w": np.array([[200]], dtype=np.int16)}
    with pytest.raises(ValueError):
        write_model(str(tmp_path / "o.bin"), graph, weights, [])


def test_corrupt_command_stream_raises_value_error(tmp_path):
    import struct as _struct

    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    weights = {"w": np.ones((8, 8), dtype=np.int8)}
    cmds = [Command(op="MATMUL", m=8, n=8, k=8)]
    path = str(tmp_path / "corrupt.bin")
    write_model(path, graph, weights, cmds)
    with open(path, "r+b") as f:
        raw = bytearray(f.read())
    (cmd_len,) = _struct.unpack_from("<I", raw, 8)
    stream_off = len(raw) - cmd_len
    count_off = stream_off + 2  # u16 magic, then u32 count
    _struct.pack_into("<I", raw, count_off, 50)
    with open(path, "wb") as f:
        f.write(bytes(raw))
    with pytest.raises(ValueError):
        read_model(path)
