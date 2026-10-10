import struct

import numpy as np
import pytest

from compiler.binary import write_model
from compiler.codegen import emit_matmul, lower_matmul
from compiler.graph import Graph, Node
from compiler.middleend import lower_graph, optimize, schedule, validate
from python.edge_npu.commands import Command
from python.edge_npu.reference import bias_add, matmul_int8, relu, requantize
from python.edge_npu.sdk import Device
from python.edge_npu.tensor import TensorSpec


def _specs(*names, dtype="int32", shape=(8, 8)):
    return {n: TensorSpec(name=n, dtype=dtype, shape=shape) for n in names}


def test_schedule_chain_preserves_order():
    matmul = Node(op="matmul", inputs=("a", "w"), output="c")
    relu = Node(op="relu", inputs=("c",), output="r")
    order = schedule(Graph(nodes=(matmul, relu)), _specs("a", "w", "c", "r"))
    assert [n.output for n in order] == ["c", "r"]


def test_schedule_reorders_reversed_tuple():
    matmul = Node(op="matmul", inputs=("a", "w"), output="c")
    relu = Node(op="relu", inputs=("c",), output="r")
    order = schedule(Graph(nodes=(relu, matmul)), _specs("a", "w", "c", "r"))
    assert [n.output for n in order] == ["c", "r"]


def test_schedule_names_cycle():
    g = Graph(nodes=(Node(op="relu", inputs=("y",), output="x"),
                     Node(op="relu", inputs=("x",), output="y")))
    with pytest.raises(ValueError, match="cycle"):
        schedule(g, _specs("x", "y"))


def test_schedule_rejects_unknown_input():
    g = Graph(nodes=(Node(op="relu", inputs=("zz",), output="r"),))
    with pytest.raises(ValueError, match="zz"):
        schedule(g, _specs("r"))


def test_schedule_rejects_output_missing_from_specs():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    with pytest.raises(ValueError, match="'c'"):
        schedule(g, _specs("a", "w"))


def test_schedule_deterministic_tie_break():
    # Tuple order is (C, A, B); A and B are both ready first, A has the lower index.
    c = Node(op="bias_add", inputs=("y", "q"), output="z")
    a = Node(op="relu", inputs=("p",), output="q")
    b = Node(op="relu", inputs=("x",), output="y")
    order = schedule(Graph(nodes=(c, a, b)), _specs("p", "q", "x", "y", "z"))
    assert [n.output for n in order] == ["q", "y", "z"]


def test_collapse_double_relu():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="r1"),
                     Node(op="relu", inputs=("r1",), output="r2")))
    out, _ = optimize(g, {"w": np.ones((8, 8), dtype=np.int8)})
    assert [n.output for n in out.nodes] == ["c", "r1"]


def test_collapse_rewires_consumers_to_surviving_relu():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="r1"),
                     Node(op="relu", inputs=("r1",), output="r2"),
                     Node(op="requantize", inputs=("r2",), output="q")))
    out, _ = optimize(g, {"w": np.ones((8, 8), dtype=np.int8)})
    assert [n.output for n in out.nodes] == ["c", "r1", "q"]
    assert out.nodes[2].inputs == ("r1",)


def test_prune_weights():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    weights = {"w": np.ones((8, 8), dtype=np.int8),
               "orphan": np.zeros((4,), dtype=np.int32)}
    out, pruned = optimize(g, weights)
    assert set(pruned) == {"w"}
    assert set(weights) == {"w", "orphan"}  # input dict untouched


def test_optimize_fixed_point():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="r1"),
                     Node(op="relu", inputs=("r1",), output="r2"),
                     Node(op="relu", inputs=("r2",), output="r3")))
    w = {"w": np.ones((8, 8), dtype=np.int8)}
    once, w1 = optimize(g, w)
    twice, w2 = optimize(once, w1)
    assert twice == once and w2.keys() == w1.keys()


def test_fanout_rejected():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("c", "b"), output="d"))
    specs = _specs("a", "w", "c", "r", "b", "d")
    order = schedule(Graph(nodes=nodes), specs)
    with pytest.raises(ValueError, match="'c'"):
        validate(order)


def test_shared_weight_is_not_fanout():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="bias_add", inputs=("c", "b"), output="d"),
             Node(op="bias_add", inputs=("d", "b"), output="e"))
    specs = _specs("a", "w", "c", "b", "d", "e")
    validate(schedule(Graph(nodes=nodes), specs))  # must not raise


def test_two_sinks_rejected():
    nodes = (Node(op="relu", inputs=("x",), output="r1"),
             Node(op="relu", inputs=("y",), output="r2"))
    specs = _specs("x", "r1", "y", "r2")
    with pytest.raises(ValueError, match="2"):
        validate(schedule(Graph(nodes=nodes), specs))


def test_unsupported_op_rejected():
    nodes = (Node(op="load", inputs=("x",), output="l"),)
    specs = _specs("x", "l")
    with pytest.raises(ValueError, match="load"):
        validate(schedule(Graph(nodes=nodes), specs))


def test_empty_graph_rejected():
    with pytest.raises(ValueError, match="0"):
        validate(schedule(Graph(nodes=()), {}))


def _chain_graph():
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


def test_emit_matmul_matches_lower_matmul():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(64, 64)),
             "w": TensorSpec(name="w", dtype="int8", shape=(64, 64)),
             "c": TensorSpec(name="c", dtype="int32", shape=(64, 64))}
    assert emit_matmul(node, specs, 0x1000, 0x2000, 0x3000) == \
        lower_matmul(node, specs, base=0x1000)


def test_lower_graph_matmul_stream_equals_lower_matmul():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(64, 64)),
             "w": TensorSpec(name="w", dtype="int8", shape=(64, 64)),
             "c": TensorSpec(name="c", dtype="int32", shape=(64, 64))}
    lg = lower_graph(Graph(nodes=(node,)), specs,
                     {"w": np.ones((64, 64), dtype=np.int8)}, {})
    assert lg.cmds == lower_matmul(node, specs, base=0x1000)
    assert lg.graph == Graph(nodes=(node,))


def test_lower_graph_chain_exact_stream():
    nodes, specs = _chain_graph()
    lg = lower_graph(Graph(nodes=nodes), specs,
                     {"w": np.ones((8, 8), dtype=np.int8),
                      "b": np.zeros((8,), dtype=np.int32)}, {})
    scale_bits = struct.unpack("<I", struct.pack("<f", 0.02))[0]
    assert lg.cmds == [
        Command(op="LOAD", address=0x1000, size=64),
        Command(op="LOAD", address=0x1040, size=64),
        Command(op="MATMUL", m=8, n=8, k=8),
        Command(op="STORE", address=0x1080, size=256),
        Command(op="REQUANTIZE", address=0x1000, size=64, reserved=scale_bits, m=128),
        Command(op="STORE", address=0x1000, size=64),
    ]


def test_lower_graph_pool_requires_attrs():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="max_pool", inputs=("c",), output="p"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 4, 4))}
    with pytest.raises(ValueError, match="'p'"):
        lower_graph(Graph(nodes=nodes), specs,
                    {"w": np.ones((8, 8), dtype=np.int8)}, {})


def test_lower_graph_pool_uses_attrs():
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="c"),
             Node(op="max_pool", inputs=("c",), output="p"))
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 10, 10)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 4, 4))}
    lg = lower_graph(Graph(nodes=nodes), specs,
                     {"w": np.ones((8, 8, 3, 3), dtype=np.int8)},
                     {"p": {"size": 2, "stride": 2}})
    assert lg.cmds == [
        Command(op="LOAD", address=0x1000, size=800),
        Command(op="LOAD", address=0x1340, size=576),
        Command(op="CONV2D", address=0x1580, size=2048, m=8, n=8, k=8),
        Command(op="MAX_POOL", address=0x1000, size=512, m=2, n=2),
        Command(op="STORE", address=0x1000, size=512),
    ]


def test_lower_graph_end_to_end_matches_reference(tmp_path):
    nodes, specs = _chain_graph()
    rng = np.random.default_rng(11)
    w = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    b = rng.integers(-100, 100, size=(8,), dtype=np.int32)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w": w, "b": b}, {})
    path = str(tmp_path / "chain.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    expected = requantize(bias_add(relu(matmul_int8(x, w)), b), 0.02)
    np.testing.assert_array_equal(out, expected)


def test_lower_graph_empty_rejected():
    with pytest.raises(ValueError, match="0"):
        lower_graph(Graph(nodes=()), {}, {}, {})
