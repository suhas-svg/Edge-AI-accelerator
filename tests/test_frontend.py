"""ONNX frontend: parse .onnx files into Graph + weights + specs."""
import os

import numpy as np
import pytest
from onnx import TensorProto, helper


def _save(model, path):
    from onnx import save_model
    save_model(model, path)


def _matmul_relu_model():
    w = np.ones((8, 8), dtype=np.float32)
    node_mm = helper.make_node("MatMul", ["a", "w"], ["c"])
    node_relu = helper.make_node("Relu", ["c"], ["y"])
    graph = helper.make_graph(
        [node_mm, node_relu], "mm-relu",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 8])],
        [helper.make_tensor("w", TensorProto.FLOAT, [8, 8], w.flatten().tolist())],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _conv_pool_model():
    w = np.ones((8, 8, 3, 3), dtype=np.float32)
    node_conv = helper.make_node("Conv", ["x", "w"], ["c"])
    node_pool = helper.make_node("MaxPool", ["c"], ["p"],
                                 kernel_shape=[2, 2], strides=[2, 2])
    graph = helper.make_graph(
        [node_conv, node_pool], "conv-pool",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 10, 10])],
        [helper.make_tensor_value_info("p", TensorProto.FLOAT, [1, 8, 4, 4])],
        [helper.make_tensor("w", TensorProto.FLOAT, [8, 8, 3, 3], w.flatten().tolist())],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_parse_matmul_relu(tmp_path):
    from compiler.frontend import load_onnx
    path = os.path.join(str(tmp_path), "mm.onnx")
    _save(_matmul_relu_model(), path)
    graph, weights, specs, attrs = load_onnx(path)
    assert [n.op for n in graph.nodes] == ["matmul", "relu"]
    assert [n.output for n in graph.nodes] == ["c", "y"]
    assert list(weights) == ["w"]
    assert weights["w"].shape == (8, 8)
    assert specs["a"].shape == (8, 8) and specs["a"].dtype == "fp32"
    assert attrs == {}


def test_parse_conv_pool_attrs(tmp_path):
    from compiler.frontend import load_onnx
    path = os.path.join(str(tmp_path), "cp.onnx")
    _save(_conv_pool_model(), path)
    graph, weights, specs, attrs = load_onnx(path)
    assert [n.op for n in graph.nodes] == ["conv2d", "max_pool"]
    assert specs["x"].shape == (8, 10, 10)
    assert specs["c"].shape == (8, 8, 8)
    assert specs["p"].shape == (8, 4, 4)
    assert attrs == {"p": {"size": 2, "stride": 2}}


def test_batch_two_rejected(tmp_path):
    from compiler.frontend import load_onnx
    w = np.ones((8, 8, 3, 3), dtype=np.float32)
    node = helper.make_node("Conv", ["x", "w"], ["c"])
    graph = helper.make_graph(
        [node], "conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8, 10, 10])],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 8, 8, 8])],
        [helper.make_tensor("w", TensorProto.FLOAT, [8, 8, 3, 3], w.flatten().tolist())],
    )
    path = os.path.join(str(tmp_path), "b2.onnx")
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    with pytest.raises(ValueError, match="batch"):
        load_onnx(path)


def test_unsupported_op_rejected(tmp_path):
    from compiler.frontend import load_onnx
    node = helper.make_node("Softmax", ["a"], ["y"], axis=-1)
    graph = helper.make_graph(
        [node], "sm",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 8])],
    )
    path = os.path.join(str(tmp_path), "sm.onnx")
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    with pytest.raises(ValueError, match="Softmax"):
        load_onnx(path)


def test_gemm_transB_rejected(tmp_path):
    from compiler.frontend import load_onnx
    w = np.ones((8, 8), dtype=np.float32)
    node = helper.make_node("Gemm", ["a", "w"], ["y"], transB=1)
    graph = helper.make_graph(
        [node], "gemm",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 8])],
        [helper.make_tensor("w", TensorProto.FLOAT, [8, 8], w.flatten().tolist())],
    )
    path = os.path.join(str(tmp_path), "gemm.onnx")
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    with pytest.raises(ValueError, match="transB"):
        load_onnx(path)


def test_conv_padded_rejected(tmp_path):
    from compiler.frontend import load_onnx
    w = np.ones((8, 8, 3, 3), dtype=np.float32)
    node = helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [node], "conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 10, 10])],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 8, 10, 10])],
        [helper.make_tensor("w", TensorProto.FLOAT, [8, 8, 3, 3], w.flatten().tolist())],
    )
    path = os.path.join(str(tmp_path), "conv.onnx")
    _save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    with pytest.raises(ValueError, match="pads"):
        load_onnx(path)
