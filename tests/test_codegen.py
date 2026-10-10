import pytest

from compiler.codegen import lower_matmul
from compiler.graph import Node
from python.edge_npu.commands import Command
from python.edge_npu.tensor import TensorSpec


def test_lower_64x64_matmul():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(64, 64)),
             "w": TensorSpec(name="w", dtype="int8", shape=(64, 64)),
             "c": TensorSpec(name="c", dtype="int32", shape=(64, 64))}
    assert lower_matmul(node, specs, base=0x1000) == [
        Command(op="LOAD", address=0x1000, size=4096),
        Command(op="LOAD", address=0x2000, size=4096),
        Command(op="MATMUL", m=64, n=64, k=64),
        Command(op="STORE", address=0x3000, size=16384),
    ]


def test_odd_dim_rejected():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(60, 64)),
             "w": TensorSpec(name="w", dtype="int8", shape=(64, 64)),
             "c": TensorSpec(name="c", dtype="int32", shape=(60, 64))}
    with pytest.raises(ValueError):
        lower_matmul(node, specs, base=0x1000)


def test_non_matmul_op_rejected():
    node = Node(op="relu", inputs=("c",), output="y")
    specs = {"c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "y": TensorSpec(name="y", dtype="int32", shape=(8, 8))}
    with pytest.raises(ValueError):
        lower_matmul(node, specs, base=0x1000)


def test_output_shape_mismatch_rejected():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(16, 16))}
    with pytest.raises(ValueError):
        lower_matmul(node, specs, base=0x1000)


def test_missing_tensor_raises_value_error():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8))}
    with pytest.raises(ValueError):
        lower_matmul(node, specs, base=0x1000)
