import pytest

from compiler.graph import Graph, Node


def test_two_node_chain_orders():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="y")))
    assert [n.output for n in g.order()] == ["c", "y"]


def test_duplicate_output_raises():
    with pytest.raises(ValueError):
        Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="c")))
