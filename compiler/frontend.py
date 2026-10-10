"""ONNX frontend: .onnx files → Graph + weights + specs + attrs.

Parser plus supported-operator validation (spec section 16). Shapes for
intermediate tensors come from ONNX shape inference. Everything the
lowering passes cannot express is a named ValueError, not a silent
reinterpretation.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

from compiler.graph import Graph, Node
from python.edge_npu.tensor import TensorSpec

_OPS = {"MatMul": "matmul", "Gemm": "matmul", "Relu": "relu", "Add": "bias_add",
        "Conv": "conv2d", "MaxPool": "max_pool"}

_DTYPES = {TensorProto.FLOAT: "fp32", TensorProto.INT8: "int8",
           TensorProto.INT32: "int32"}
_NP_DTYPES = {np.dtype("float32"): "fp32", np.dtype("int8"): "int8",
              np.dtype("int32"): "int32"}


def _attrs(node: onnx.NodeProto) -> dict:
    out = {}
    for a in node.attribute:
        v = helper.get_attribute_value(a)
        out[a.name] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def _shape(elem) -> tuple[int, ...]:
    dims = []
    for d in elem.type.tensor_type.shape.dim:
        if d.dim_value <= 0:
            raise ValueError(f"tensor {elem.name!r} has dynamic shape")
        dims.append(d.dim_value)
    return tuple(dims)


def _squeeze_batch(spec: TensorSpec) -> TensorSpec:
    """ONNX vision tensors are NCHW; the NPU interior is CHW. Squeeze N=1."""
    if len(spec.shape) == 4:
        if spec.shape[0] != 1:
            raise ValueError(f"tensor {spec.name!r} batch {spec.shape[0]} unsupported")
        return TensorSpec(name=spec.name, dtype=spec.dtype, shape=spec.shape[1:],
                          scale=spec.scale, zero_point=spec.zero_point)
    return spec


def load_onnx(path: str) -> tuple[Graph, dict[str, np.ndarray],
                                  dict[str, TensorSpec], dict[str, dict]]:
    try:
        model = onnx.load(path)
    except Exception as e:
        raise ValueError(f"cannot load ONNX model {path!r}: {e}")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        raise ValueError(f"ONNX shape inference failed for {path!r}: {e}")
    graph = model.graph

    weights: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype not in _NP_DTYPES:
            raise ValueError(f"initializer {init.name!r} dtype {arr.dtype} unsupported")
        weights[init.name] = arr

    table: dict[str, TensorSpec] = {}
    for elem in list(graph.input) + list(graph.output) + list(graph.value_info):
        t = elem.type.tensor_type
        if t.elem_type not in _DTYPES:
            raise ValueError(f"tensor {elem.name!r} dtype {t.elem_type} unsupported")
        table[elem.name] = TensorSpec(name=elem.name, dtype=_DTYPES[t.elem_type],
                                      shape=_shape(elem))
    for name, arr in weights.items():
        table[name] = TensorSpec(name=name, dtype=_NP_DTYPES[arr.dtype],
                                 shape=tuple(arr.shape))
    for name in list(table):
        if name not in weights:
            table[name] = _squeeze_batch(table[name])

    for node in graph.node:
        for name in list(node.input) + list(node.output):
            if name != "" and name not in table:
                raise ValueError(
                    f"tensor {name!r} has no shape (shape inference gap)")

    nodes: list[Node] = []
    attrs: dict[str, dict] = {}
    for node in graph.node:
        if node.op_type not in _OPS:
            raise ValueError(f"unsupported ONNX op {node.op_type!r}")
        if len(node.output) != 1:
            raise ValueError(f"op {node.op_type!r} must have 1 output")
        a = _attrs(node)
        _validate_attrs(node.op_type, a)
        op = _OPS[node.op_type]
        inputs = tuple(i for i in node.input if i != "")
        if op == "max_pool":
            kernel = a.get("kernel_shape", [2, 2])
            if len(kernel) != 2 or kernel[0] != kernel[1]:
                raise ValueError(f"MaxPool kernel must be square, got {kernel}")
            stride = a.get("strides", kernel)
            if len(stride) != 2 or stride[0] != stride[1]:
                raise ValueError(f"MaxPool stride must be square, got {stride}")
            attrs[node.output[0]] = {"size": int(kernel[0]), "stride": int(stride[0])}
        nodes.append(Node(op=op, inputs=inputs, output=node.output[0]))
    return Graph(nodes=tuple(nodes)), weights, table, attrs


def _validate_attrs(op_type: str, a: dict) -> None:
    if op_type == "Gemm":
        if a.get("transB", 0) != 0:
            raise ValueError("Gemm transB=1 unsupported")
        if a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 0.0:
            raise ValueError("Gemm only supports alpha=1 beta=0")
    elif op_type == "Conv":
        if a.get("strides", [1, 1]) != [1, 1]:
            raise ValueError(f"Conv strides {a['strides']} unsupported (need [1, 1])")
        if a.get("pads", [0, 0, 0, 0]) != [0, 0, 0, 0]:
            raise ValueError(f"Conv pads {a['pads']} unsupported (need valid padding)")
        if a.get("dilations", [1, 1]) != [1, 1]:
            raise ValueError(f"Conv dilations {a['dilations']} unsupported")
        if a.get("group", 1) != 1:
            raise ValueError(f"Conv group {a['group']} unsupported")
