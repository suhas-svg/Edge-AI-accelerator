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


def test_lower_relu():
    from compiler.codegen import lower_relu
    node = Node(op="relu", inputs=("c",), output="y")
    specs = {"c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "y": TensorSpec(name="y", dtype="int32", shape=(8, 8))}
    assert lower_relu(node, specs, base=0x3000) == [
        Command(op="RELU", address=0x3000, size=256),
    ]


def test_lower_bias_add():
    from compiler.codegen import lower_bias_add
    node = Node(op="bias_add", inputs=("c", "b"), output="y")
    specs = {"c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(8,)),
             "y": TensorSpec(name="y", dtype="int32", shape=(8, 8))}
    assert lower_bias_add(node, specs, base=0x3000) == [
        Command(op="BIAS_ADD", address=0x3000, size=256),
    ]


def test_relu_odd_elements_rejected():
    from compiler.codegen import lower_relu
    node = Node(op="relu", inputs=("c",), output="y")
    specs = {"c": TensorSpec(name="c", dtype="int32", shape=(7, 8)),
             "y": TensorSpec(name="y", dtype="int32", shape=(7, 8))}
    with pytest.raises(ValueError):
        lower_relu(node, specs, base=0x3000)


def test_bias_incompatible_shape_rejected():
    from compiler.codegen import lower_bias_add
    node = Node(op="bias_add", inputs=("c", "b"), output="y")
    specs = {"c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(7,)),
             "y": TensorSpec(name="y", dtype="int32", shape=(8, 8))}
    with pytest.raises(ValueError):
        lower_bias_add(node, specs, base=0x3000)


def _conv_specs():
    return {"x": TensorSpec(name="x", dtype="int8", shape=(8, 10, 10)),
            "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
            "y": TensorSpec(name="y", dtype="int32", shape=(8, 8, 8))}


def test_lower_conv2d():
    from compiler.codegen import lower_conv2d
    node = Node(op="conv2d", inputs=("x", "w"), output="y")
    assert lower_conv2d(node, _conv_specs(), base=0x3000) == [
        Command(op="CONV2D", address=0x3000, size=2048, m=8, n=8, k=8),
    ]


def test_conv_channel_mismatch_rejected():
    from compiler.codegen import lower_conv2d
    node = Node(op="conv2d", inputs=("x", "w"), output="y")
    specs = _conv_specs()
    specs["w"] = TensorSpec(name="w", dtype="int8", shape=(8, 4, 3, 3))
    with pytest.raises(ValueError):
        lower_conv2d(node, specs, base=0x3000)


def test_conv_odd_dim_rejected():
    from compiler.codegen import lower_conv2d
    node = Node(op="conv2d", inputs=("x", "w"), output="y")
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 9, 9)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "y": TensorSpec(name="y", dtype="int32", shape=(8, 7, 7))}
    with pytest.raises(ValueError):
        lower_conv2d(node, specs, base=0x3000)


def test_lower_max_pool():
    from compiler.codegen import lower_max_pool
    node = Node(op="max_pool", inputs=("y",), output="p")
    specs = {"y": TensorSpec(name="y", dtype="int32", shape=(8, 8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 4, 4))}
    assert lower_max_pool(node, specs, base=0x4000, size=2, stride=2) == [
        Command(op="MAX_POOL", address=0x4000, size=512, m=2, n=2),
    ]


def test_pool_shape_mismatch_rejected():
    from compiler.codegen import lower_max_pool
    node = Node(op="max_pool", inputs=("y",), output="p")
    specs = {"y": TensorSpec(name="y", dtype="int32", shape=(8, 8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 3, 3))}
    with pytest.raises(ValueError):
        lower_max_pool(node, specs, base=0x4000, size=2, stride=2)


def _requant_specs():
    return {"c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
            "q": TensorSpec(name="q", dtype="int8", shape=(8, 8))}


def test_lower_requantize():
    import struct
    from compiler.codegen import lower_requantize
    node = Node(op="requantize", inputs=("c",), output="q")
    cmds = lower_requantize(node, _requant_specs(), base=0x5000, scale=0.05)
    assert len(cmds) == 1
    c = cmds[0]
    assert c.op == "REQUANTIZE" and c.address == 0x5000 and c.size == 64
    assert struct.unpack("<f", struct.pack("<I", c.reserved))[0] == pytest.approx(0.05)
    assert c.m == 128 and c.n == 0 and c.k == 0


def test_requantize_bad_scale_rejected():
    from compiler.codegen import lower_requantize
    node = Node(op="requantize", inputs=("c",), output="q")
    with pytest.raises(ValueError):
        lower_requantize(node, _requant_specs(), base=0x5000, scale=0.0)


def test_requantize_bad_zp_rejected():
    from compiler.codegen import lower_requantize
    node = Node(op="requantize", inputs=("c",), output="q")
    with pytest.raises(ValueError):
        lower_requantize(node, _requant_specs(), base=0x5000, scale=0.05,
                         zero_point=200)


def test_requantize_dtype_mismatch_rejected():
    from compiler.codegen import lower_requantize
    node = Node(op="requantize", inputs=("c",), output="q")
    specs = {"c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "q": TensorSpec(name="q", dtype="int32", shape=(8, 8))}
    with pytest.raises(ValueError):
        lower_requantize(node, specs, base=0x5000, scale=0.05)