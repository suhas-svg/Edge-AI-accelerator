"""Behavior tests. Each fails for a real numerics defect."""
import numpy as np
import pytest
from python.edge_npu.reference import (matmul_fp32, quantize_int8, dequantize_int8,
                                       matmul_int8, quantization_error)


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


def test_quantization_error_is_bounded_by_half_a_step():
    """Spec section 18: compare FP32 against INT8 and calculate the error."""
    x = np.array([0.03, -0.04, 0.07, 1.0], dtype=np.float32)
    scale = 0.05
    q = quantize_int8(x, scale=scale)
    deq = dequantize_int8(q, scale=scale)
    assert quantization_error(x, deq) <= scale / 2 + 1e-6


def test_int32_overflow_is_rejected_not_wrapped():
    """A result outside INT32 range must raise, not silently wrap.

    K = 200000 makes the true dot product -3,251,200,000, which int32
    cannot hold. Without the guard numpy wraps it to +1,043,767,296.
    """
    k = 200000
    a = np.full((1, k), -128, dtype=np.int8)
    b = np.full((k, 1), 127, dtype=np.int8)
    with pytest.raises(ValueError, match="INT32"):
        matmul_int8(a, b)


def test_quantization_error_is_zero_when_exact():
    x = np.array([0.05, 0.10], dtype=np.float32)
    assert quantization_error(x, dequantize_int8(quantize_int8(x, 0.05), 0.05)) == 0.0
