import pytest

from compiler.graph import Graph, Node
from compiler.middleend import legalize, lower_graph, schedule
from python.edge_npu.tensor import TensorSpec


def _pool_chain():
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="c"),
             Node(op="max_pool", inputs=("c",), output="p"))
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 10, 10)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 4, 4))}
    return nodes, specs


def test_legalize_malformed_pool_attrs_named():
    nodes, specs = _pool_chain()
    order = schedule(Graph(nodes=nodes), specs)
    with pytest.raises(ValueError, match="bad attrs"):
        legalize(order, specs, {"p": {}})


def test_lower_graph_malformed_pool_attrs_named():
    import numpy as np
    nodes, specs = _pool_chain()
    with pytest.raises(ValueError, match="bad attrs"):
        lower_graph(Graph(nodes=nodes), specs,
                    {"w": np.ones((8, 8, 3, 3), dtype=np.int8)},
                    {"p": {"size": 2}})


def test_lower_graph_passes_zero_point_through():
    import numpy as np
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="requantize", inputs=("c",), output="q"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "q": TensorSpec(name="q", dtype="int8", shape=(8, 8),
                             scale=0.02, zero_point=5)}
    lg = lower_graph(Graph(nodes=nodes), specs,
                     {"w": np.ones((8, 8), dtype=np.int8)}, {})
    rq = [c for c in lg.cmds if c.op == "REQUANTIZE"]
    assert len(rq) == 1 and rq[0].m == 133


def test_dtype_itemsize():
    from python.edge_npu.tensor import dtype_itemsize
    assert (dtype_itemsize("fp32"), dtype_itemsize("int8"),
            dtype_itemsize("int32")) == (4, 1, 4)
    with pytest.raises(ValueError):
        dtype_itemsize("int4")


def test_peak_uses_logical_bytes_not_spans():
    from compiler.memory import MemoryConfig, plan_memory
    order = [Node(op="matmul", inputs=("a", "w"), output="c")]
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(10, 10)),
             "w": TensorSpec(name="w", dtype="int8", shape=(10, 10)),
             "c": TensorSpec(name="c", dtype="int32", shape=(10, 10))}
    plan = plan_memory(order, specs, {}, MemoryConfig())
    assert plan.slots["a"].address == 0x1000
    assert plan.slots["w"].address == 0x1080
    assert plan.slots["c"].address == 0x1100
    assert plan.peak_bytes == 0x290
