"""Golden reference. Pure Python + numpy. This is the truth RTL is checked against."""
from __future__ import annotations
import numpy as np


def matmul_fp32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) @ b.astype(np.float32)


def quantize_int8(x: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
    q = np.round(x / scale).astype(np.int32) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_int8(q: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


def matmul_int8(a_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """INT8 in, INT32 accum out. Matches the 8x8 MAC array contract (spec section 5)."""
    return (a_q.astype(np.int32) @ b_q.astype(np.int32)).astype(np.int32)


def quantization_error(fp32_out: np.ndarray, deq_out: np.ndarray) -> float:
    """Max absolute error between an FP32 result and its dequantized INT8 twin.

    Spec section 18 requires the compiler to compare FP32 against INT8 and
    calculate quantization error. This is that calculation.
    """
    return float(np.max(np.abs(fp32_out - deq_out)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def bias_add(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x + b


def conv2d_int8(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Naive valid-padding stride-1 conv. INT8 in, INT32 accum out."""
    c, h, wdt = x.shape
    k, kc, kh, kw = w.shape
    assert kc == c, f"channel mismatch {kc} vs {c}"
    oh, ow = h - kh + 1, wdt - kw + 1
    out = np.zeros((k, oh, ow), dtype=np.int32)
    for f in range(k):
        for i in range(oh):
            for j in range(ow):
                patch = x[:, i:i + kh, j:j + kw].astype(np.int32)
                out[f, i, j] = np.sum(patch * w[f].astype(np.int32))
    return out


def max_pool(x: np.ndarray, size: int, stride: int) -> np.ndarray:
    c, h, wdt = x.shape
    oh, ow = (h - size) // stride + 1, (wdt - size) // stride + 1
    out = np.zeros((c, oh, ow), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            window = x[:, i * stride:i * stride + size, j * stride:j * stride + size]
            out[:, i, j] = np.max(window, axis=(1, 2))
    return out


def requantize(acc: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
    """Map INT32 accumulator to INT8. ``scale`` is output-per-accumulator,
    so this multiplies. ``quantize_int8`` takes the opposite convention
    (FP32 value per INT8 step, so it divides); see docs/tensor-format.md.
    """
    q = np.round(acc.astype(np.float32) * scale).astype(np.int32) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)
