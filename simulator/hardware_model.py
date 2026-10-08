"""Cycle-counting stand-in for the RTL MAC array. Same numerics as reference."""
from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.edge_npu.reference import matmul_int8


def run_matmul(a_q: np.ndarray, b_q: np.ndarray) -> tuple[np.ndarray, int]:
    out = matmul_int8(a_q, b_q)
    m, n, k = a_q.shape[0], b_q.shape[1], a_q.shape[1]
    cycles = (m * n * k) // 64  # 8x8 array retires 64 MACs per cycle
    return out, cycles
