import numpy as np

from compiler.binary import write_model
from compiler.graph import Graph, Node
from python.edge_npu.commands import Command
from python.edge_npu.sdk import Device


def _tiny_model(path):
    w = np.ones((8, 8), dtype=np.int8)
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    cmds = [Command(op="LOAD", address=0x1000, size=64),
            Command(op="LOAD", address=0x1040, size=64),
            Command(op="MATMUL", m=8, n=8, k=8),
            Command(op="STORE", address=0x1080, size=256)]
    write_model(path, graph, {"w": w}, cmds)


def test_matmul_counters(tmp_path):
    path = str(tmp_path / "tiny.bin")
    _tiny_model(path)
    model = Device().load_model(path)
    model.predict(np.ones((8, 8), dtype=np.int8))
    stats = model.get_stats()
    assert stats["cycles"] == 8
    assert stats["busy_cycles"] == 8
    assert stats["idle_cycles"] == 0
    assert stats["dma_bytes"] == 64 + 64 + 256
    assert stats["command_counts"] == {"LOAD": 2, "MATMUL": 1, "STORE": 1}


def test_chain_counters(tmp_path):
    from compiler.codegen import lower_bias_add, lower_matmul, lower_requantize
    from python.edge_npu.tensor import TensorSpec
    rng = np.random.default_rng(2)
    w = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="bias_add", inputs=("c", "b"), output="d"),
             Node(op="requantize", inputs=("d",), output="q"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(8,)),
             "d": TensorSpec(name="d", dtype="int32", shape=(8, 8)),
             "q": TensorSpec(name="q", dtype="int8", shape=(8, 8), scale=0.02)}
    b = rng.integers(-100, 100, size=(8,), dtype=np.int32)
    cmds = (lower_matmul(nodes[0], specs, base=0x1000)
            + lower_bias_add(nodes[1], specs, base=0x3000)
            + lower_requantize(nodes[2], specs, base=0x5000, scale=0.02)
            + [Command(op="STORE", address=0x6000, size=64)])
    path = str(tmp_path / "chain.bin")
    write_model(path, Graph(nodes=nodes), {"w": w, "b": b}, cmds)
    model = Device().load_model(path)
    model.predict(rng.integers(-128, 127, size=(8, 8), dtype=np.int8))
    stats = model.get_stats()
    assert stats["busy_cycles"] == 8
    assert stats["idle_cycles"] == 0
    assert stats["dma_bytes"] == 64 + 64 + 256 + 64
    assert stats["command_counts"] == {"LOAD": 2, "MATMUL": 1, "STORE": 2,
                                       "BIAS_ADD": 1, "REQUANTIZE": 1}
