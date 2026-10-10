import numpy as np
import pytest

from compiler.graph import Graph, Node
from compiler.middleend import optimize, schedule
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
