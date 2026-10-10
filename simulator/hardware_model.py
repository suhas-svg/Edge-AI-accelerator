"""Cycle-counting stand-in for the RTL MAC array.

Numerics are computed independently of ``python.edge_npu.reference`` so that
``tests/test_simulator.py`` is a real cross-check rather than a function
compared against its own output.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MAC_ARRAY = 8
MACS_PER_CYCLE = MAC_ARRAY * MAC_ARRAY


def mac_array_matmul(a_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """INT8 matmul accumulated in INT32, written as an explicit MAC loop.

    A tiled double loop over an 8x8 MAC tile, so it mirrors what the
    hardware does instead of calling the reference implementation.
    """
    m, k = a_q.shape
    _, n = b_q.shape
    a32 = a_q.astype(np.int32)
    b32 = b_q.astype(np.int32)
    out = np.zeros((m, n), dtype=np.int32)
    for i0 in range(0, m, MAC_ARRAY):
        for j0 in range(0, n, MAC_ARRAY):
            acc = np.zeros((MAC_ARRAY, MAC_ARRAY), dtype=np.int32)
            for p0 in range(0, k, MAC_ARRAY):
                a_tile = a32[i0:i0 + MAC_ARRAY, p0:p0 + MAC_ARRAY]
                b_tile = b32[p0:p0 + MAC_ARRAY, j0:j0 + MAC_ARRAY]
                acc += a_tile @ b_tile
            rows = min(MAC_ARRAY, m - i0)
            cols = min(MAC_ARRAY, n - j0)
            out[i0:i0 + rows, j0:j0 + cols] = acc[:rows, :cols]
    return out


def run_matmul(a_q: np.ndarray, b_q: np.ndarray) -> tuple[np.ndarray, int]:
    out = mac_array_matmul(a_q, b_q)
    m, n, k = a_q.shape[0], b_q.shape[1], a_q.shape[1]
    cycles = (m * n * k) // MACS_PER_CYCLE
    return out, cycles


def mac_array_conv2d(x_q: np.ndarray, w_q: np.ndarray) -> np.ndarray:
    """INT8 valid-padding stride-1 conv accumulated in INT32.

    Output-stationary scalar MAC loop, independent of the reference
    implementation so the parity test is a real cross-check.
    """
    c, h, wd = x_q.shape
    k, kc, kh, kw = w_q.shape
    if kc != c:
        raise ValueError(f"channel mismatch {kc} vs {c}")
    x32 = x_q.astype(np.int32)
    w32 = w_q.astype(np.int32)
    oh, ow = h - kh + 1, wd - kw + 1
    out = np.zeros((k, oh, ow), dtype=np.int32)
    for f in range(k):
        for i in range(oh):
            for j in range(ow):
                acc = np.int32(0)
                for cc in range(c):
                    for ki in range(kh):
                        for kj in range(kw):
                            acc += x32[cc, i + ki, j + kj] * w32[f, cc, ki, kj]
                out[f, i, j] = acc
    return out


def run_conv2d(x_q: np.ndarray, w_q: np.ndarray) -> tuple[np.ndarray, int]:
    out = mac_array_conv2d(x_q, w_q)
    k, oh, ow = out.shape
    c, kh, kw = w_q.shape[1], w_q.shape[2], w_q.shape[3]
    cycles = (k * oh * ow * c * kh * kw) // MACS_PER_CYCLE
    return out, cycles
