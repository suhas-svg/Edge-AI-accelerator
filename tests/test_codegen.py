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
