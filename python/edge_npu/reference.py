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
    return float(np.max(np.abs(fp32_out - deq_out)))
