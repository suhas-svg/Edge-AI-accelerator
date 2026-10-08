"""Simulator parity. Fails for any numerics or cycle-count defect."""
import glob, os
import numpy as np
from simulator.hardware_model import run_matmul

VEC_DIR = os.path.join(os.path.dirname(__file__), "vectors")


def _vectors() -> list[str]:
    return sorted(glob.glob(os.path.join(VEC_DIR, "matmul_*x*.npz")))


def test_vectors_present():
    assert len(_vectors()) == 4, "run tools/gen_vectors.py first"


def test_sim_matches_reference_on_all_vectors():
    for path in _vectors():
        d = np.load(path)
        out, cycles = run_matmul(d["a_q"], d["b_q"])
        assert out.tolist() == d["expected"].tolist(), path
        m, n = d["expected"].shape
        k = d["a_q"].shape[1]
        assert cycles == (m * n * k) // 64, path
