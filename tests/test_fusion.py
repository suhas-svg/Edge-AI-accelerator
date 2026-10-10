import struct

import numpy as np
import pytest

from compiler.binary import write_model
from compiler.graph import Graph, Node
from compiler.middleend import fusable, lower_graph, schedule
from python.edge_npu.commands import Command
from python.edge_npu.reference import bias_add, matmul_int8, relu, requantize
from python.edge_npu.sdk import Device
from python.edge_npu.tensor import TensorSpec


def _chain():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("r", "b"), output="d"),
             Node(op="requantize", inputs=("d",), output="q"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(8,)),
             "d": TensorSpec(name="d", dtype="int32", shape=(8, 8)),
             "q": TensorSpec(name="q", dtype="int8", shape=(8, 8), scale=0.02)}
    return nodes, specs


def test_fusable_marks_relu_bias_after_head():
    nodes, specs = _chain()
    order = schedule(Graph(nodes=nodes), specs)
    assert fusable(order) == {"r", "d"}


def test_fusable_excludes_pool_and_requantize():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="max_pool", inputs=("c",), output="p"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 4, 4))}
    order = schedule(Graph(nodes=nodes), specs)
    assert fusable(order) == set()


def test_fused_stream_drops_elementwise():
    nodes, specs = _chain()
    lg = lower_graph(Graph(nodes=nodes), specs,
                     {"w": np.ones((8, 8), dtype=np.int8),
                      "b": np.zeros((8,), dtype=np.int32)}, {})
    scale_bits = struct.unpack("<I", struct.pack("<f", 0.02))[0]
    assert lg.cmds == [
        Command(op="LOAD", address=0x1000, size=64),
        Command(op="LOAD", address=0x1040, size=64),
        Command(op="MATMUL", m=8, n=8, k=8),
        Command(op="STORE", address=0x1080, size=256),
        Command(op="REQUANTIZE", address=0x1000, size=64,
                reserved=scale_bits, m=128),
        Command(op="STORE", address=0x1000, size=64),
    ]
    assert lg.graph.nodes[-1].output == "q"


def test_fused_chain_bit_exact(tmp_path):
    nodes, specs = _chain()
    rng = np.random.default_rng(31)
    w = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    b = rng.integers(-100, 100, size=(8,), dtype=np.int32)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w": w, "b": b}, {})
    path = str(tmp_path / "fused.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    expected = requantize(bias_add(relu(matmul_int8(x, w)), b), 0.02)
    np.testing.assert_array_equal(out, expected)


def test_banks1_preserves_layout():
    from compiler.memory import MemoryConfig, plan_memory
    order = [Node(op="matmul", inputs=("a", "w"), output="c")]
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8))}
    assert (plan_memory(order, specs, {}, MemoryConfig())
            == plan_memory(order, specs, {}, MemoryConfig(banks=1)))


def test_banks_must_be_1_or_2():
    from compiler.memory import MemoryConfig
    with pytest.raises(ValueError):
        MemoryConfig(banks=0)
    with pytest.raises(ValueError):
        MemoryConfig(banks=3)


def test_banks2_places_weights_high():
    from compiler.memory import MemoryConfig, plan_memory
    order = [Node(op="matmul", inputs=("a", "w"), output="c")]
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8))}
    plan = plan_memory(order, specs, {}, MemoryConfig(size=0x1000, banks=2),
                       weights={"w"})
    got = {n: (s.address, s.size) for n, s in plan.slots.items()}
    assert got == {"a": (0x1000, 64), "w": (0x1FC0, 64), "c": (0x1040, 256)}
    assert plan.peak_bytes == 0x1000
