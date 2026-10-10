import pytest

from compiler.graph import Graph, Node
from compiler.middleend import schedule
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
