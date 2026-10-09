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
