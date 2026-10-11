import numpy as np

from compiler.binary import write_model
from compiler.graph import Graph, Node
from compiler.memory import MemoryConfig
from compiler.middleend import fusable, lower_graph, schedule
from python.edge_npu.reference import matmul_int8, relu
from python.edge_npu.sdk import Device
from python.edge_npu.tensor import TensorSpec


def _mlp():
    nodes = (Node(op="matmul", inputs=("a", "w1"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="matmul", inputs=("r", "w2"), output="d"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w1": TensorSpec(name="w1", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8)),
             "w2": TensorSpec(name="w2", dtype="int8", shape=(8, 8)),
             "d": TensorSpec(name="d", dtype="int32", shape=(8, 8))}
    return nodes, specs


def test_source_loads_hoisted_ahead_of_first_compute():
    nodes, specs = _mlp()
    rng = np.random.default_rng(41)
    w1 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    w2 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w1": w1, "w2": w2}, {})
    first_compute = next(i for i, c in enumerate(lg.cmds)
                         if c.op in ("MATMUL", "CONV2D"))
    assert [c.op for c in lg.cmds[:first_compute]] == ["LOAD", "LOAD", "LOAD"]
    assert [c.op for c in lg.cmds] == ["LOAD", "LOAD", "LOAD",
                                       "MATMUL", "STORE",
                                       "MATMUL", "STORE"]


def test_banks2_parks_weights_above_activations():
    nodes, specs = _mlp()
    rng = np.random.default_rng(41)
    w1 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    w2 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w1": w1, "w2": w2}, {},
                     memory=MemoryConfig(banks=2))
    slots = lg.plan.slots
    assert min(slots[w].address for w in ("w1", "w2")) > \
        max(slots[a].address for a in ("a", "c", "r", "d"))


def test_fusion_covers_relu_and_bias_tails():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("r", "b"), output="d"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(8,)),
             "d": TensorSpec(name="d", dtype="int32", shape=(8, 8))}
    order = schedule(Graph(nodes=nodes), specs)
    assert fusable(order) == {"r", "d"}


def test_hoisted_stream_still_predicts_bit_exact(tmp_path):
    nodes, specs = _mlp()
    rng = np.random.default_rng(42)
    w1 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    w2 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w1": w1, "w2": w2}, {})
    path = str(tmp_path / "hoisted.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    model = Device().load_model(path)
    out = model.predict(x)
    np.testing.assert_array_equal(out, matmul_int8(relu(matmul_int8(x, w1)), w2))
