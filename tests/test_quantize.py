"""Quantization pass: exact scales, rejection paths, and the frontend bridge."""
import numpy as np
import pytest


def test_choose_scale_exact():
    from compiler.quantize import choose_scale
    assert choose_scale(np.array([-1.0, 0.5], dtype=np.float32)) == pytest.approx(1.0 / 127)


def test_choose_scale_zeros():
    from compiler.quantize import choose_scale
    assert choose_scale(np.zeros((4,), dtype=np.float32)) == 1.0


def test_quantize_weights_exact():
    from compiler.quantize import quantize_weights
    w = np.array([[127.0, -127.0]], dtype=np.float32)
    qw, scales = quantize_weights({"w": w})
    assert qw["w"].dtype == np.int8
    assert qw["w"].tolist() == [[127, -127]]
    assert scales["w"] == pytest.approx(1.0)


def test_quantize_weights_rejects_int():
    from compiler.quantize import quantize_weights
    with pytest.raises(ValueError, match="w"):
        quantize_weights({"w": np.ones((2,), dtype=np.int32)})


def test_quantize_activations_from_samples():
    from compiler.quantize import quantize_activations
    s = quantize_activations({"a": [np.full((2,), 2.0, dtype=np.float32),
                                    np.full((2,), -4.0, dtype=np.float32)]})
    assert s["a"] == pytest.approx(4.0 / 127)


def test_quantize_activations_empty_rejected():
    from compiler.quantize import quantize_activations
    with pytest.raises(ValueError):
        quantize_activations({"a": []})


def test_quantize_specs_bridge_to_lowering():
    """Frontend FP32 specs → INT8 specs → a valid matmul command stream."""
    from compiler.codegen import lower_matmul
    from compiler.graph import Graph, Node
    from compiler.quantize import quantize_activations, quantize_specs, quantize_weights
    from python.edge_npu.commands import Command
    from python.edge_npu.tensor import TensorSpec
    rng = np.random.default_rng(8)
    wf = (rng.standard_normal((8, 8)) * 2).astype(np.float32)
    specs = {"a": TensorSpec(name="a", dtype="fp32", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="fp32", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="fp32", shape=(8, 8))}
    qw, wscales = quantize_weights({"w": wf})
    ascales = quantize_activations({"a": [rng.standard_normal((8, 8)).astype(np.float32)]})
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    qspecs = quantize_specs(Graph(nodes=(node,)), specs, {**wscales, **ascales})
    assert qspecs["w"].dtype == "int8" and qspecs["a"].dtype == "int8"
    assert qspecs["c"].dtype == "int32"
    cmds = lower_matmul(node, qspecs, base=0x1000)
    assert cmds[1] == Command(op="LOAD", address=0x1000 + 64, size=64)


def test_quantize_specs_missing_scale_rejected():
    from compiler.graph import Graph, Node
    from compiler.quantize import quantize_specs
    from python.edge_npu.tensor import TensorSpec
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="fp32", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="fp32", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="fp32", shape=(8, 8))}
    with pytest.raises(ValueError, match="a"):
        quantize_specs(Graph(nodes=(node,)), specs, {"w": 0.05})
