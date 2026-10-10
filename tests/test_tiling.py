import numpy as np
import pytest

from compiler.graph import Graph, Node
from compiler.middleend import legalize, schedule
from python.edge_npu.tensor import TensorSpec


def _mm_specs(a, w, c):
    return {"a": TensorSpec(name="a", dtype="int8", shape=a),
            "w": TensorSpec(name="w", dtype="int8", shape=w),
            "c": TensorSpec(name="c", dtype="int32", shape=c)}


def _order(nodes, specs):
    return schedule(Graph(nodes=nodes), specs)


def test_legalize_aligned_is_identity():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),)
    specs = _mm_specs((64, 64), (64, 64), (64, 64))
    assert legalize(_order(nodes, specs), specs, {}) == specs


def test_legalize_odd_matmul_pads_to_tiles():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),)
    specs = _mm_specs((60, 64), (64, 48), (60, 48))
    got = legalize(_order(nodes, specs), specs, {})
    assert got["a"].shape == (64, 64)
    assert got["w"].shape == (64, 48)
    assert got["c"].shape == (64, 48)
    assert (got["a"].dtype, got["c"].dtype) == ("int8", "int32")


def test_legalize_inner_mismatch_still_rejected():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),)
    specs = _mm_specs((8, 7), (9, 8), (8, 8))
    with pytest.raises(ValueError, match="inner dim mismatch 7 vs 9"):
        legalize(_order(nodes, specs), specs, {})


def test_legalize_conv_pads_output_spatial():
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="y"),)
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 9, 9)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "y": TensorSpec(name="y", dtype="int32", shape=(8, 7, 7))}
    got = legalize(_order(nodes, specs), specs, {})
    assert got["x"].shape == (8, 10, 10)
    assert got["w"].shape == (8, 8, 3, 3)
    assert got["y"].shape == (8, 8, 8)


def test_legalize_elementwise_inherits_padded_shape():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"))
    specs = dict(_mm_specs((60, 64), (64, 64), (60, 64)),
                 r=TensorSpec(name="r", dtype="int32", shape=(60, 64)))
    got = legalize(_order(nodes, specs), specs, {})
    assert got["c"].shape == got["r"].shape == (64, 64)


def test_legalize_pads_bias_to_padded_last_dim():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="bias_add", inputs=("c", "b"), output="d"))
    specs = dict(_mm_specs((60, 60), (60, 60), (60, 60)),
                 b=TensorSpec(name="b", dtype="int32", shape=(60,)),
                 d=TensorSpec(name="d", dtype="int32", shape=(60, 60)))
    got = legalize(_order(nodes, specs), specs, {})
    assert got["d"].shape == (64, 64)
    assert got["b"].shape == (64,)


def test_legalize_pool_derives_both_shapes():
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="c"),
             Node(op="max_pool", inputs=("c",), output="p"))
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 9, 9)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 7, 7)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 3, 3))}
    got = legalize(_order(nodes, specs), specs, {"p": {"size": 2, "stride": 2}})
    assert got["c"].shape == (8, 8, 8)
    assert got["p"].shape == (8, 4, 4)


def test_legalize_pool_needs_attrs():
    nodes = (Node(op="max_pool", inputs=("x",), output="p"),)
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 8, 8)),
             "p": TensorSpec(name="p", dtype="int8", shape=(8, 4, 4))}
    with pytest.raises(ValueError, match="'p'"):
        legalize(_order(nodes, specs), specs, {})
