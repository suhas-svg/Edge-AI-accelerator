"""Behavior tests. Each fails for a real numerics defect."""
import numpy as np
from python.edge_npu.reference import matmul_fp32, quantize_int8, dequantize_int8, matmul_int8


def test_fp32_matmul_small():
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    assert matmul_fp32(a, b).tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_int8_matmul_accumulates_to_int32():
    a = np.array([[100, 100]], dtype=np.int8)
    b = np.array([[100], [100]], dtype=np.int8)
    out = matmul_int8(a, b)
    assert out.dtype == np.int32
    assert out.tolist() == [[20000]]


def test_quantize_roundtrip_bounded():
    x = np.array([0.0, 0.05, -0.05, 1.0], dtype=np.float32)
    q = quantize_int8(x, scale=0.05)
    assert q.tolist() == [0, 1, -1, 20]
    got = dequantize_int8(q, scale=0.05)
    assert np.allclose(got, [0.0, 0.05, -0.05, 1.0], atol=1e-6)
