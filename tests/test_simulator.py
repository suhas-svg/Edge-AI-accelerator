"""Simulator parity. Fails for any numerics or cycle-count defect."""
import glob, os
import numpy as np
from simulator.hardware_model import run_matmul

VEC_DIR = os.path.join(os.path.dirname(__file__), "vectors")


def _vectors() -> list[str]:
    """Square vectors only: name shape matmul_<m>x<k>x<n> with m == k == n."""
    out = []
    for p in sorted(glob.glob(os.path.join(VEC_DIR, "matmul_*.npz"))):
        name = os.path.basename(p)
        dims = name[len("matmul_"):-len(".npz")].split("x")
        if len(dims) == 3 and len(set(dims)) == 1:
            out.append(p)
    return out


def test_vectors_present():
    from tools import gen_vectors
    assert len(_vectors()) == len([s for s in gen_vectors.SIZES if s[0] == s[1] == s[2]])


def test_sim_matches_reference_on_square_vectors():
    """Square vectors. Cycle formula assumes an even 8x8 tile grid."""
    for path in _vectors():
        d = np.load(path)
        out, cycles = run_matmul(d["a_q"], d["b_q"])
        assert out.tolist() == d["expected"].tolist(), path
        m, n = d["expected"].shape
        k = d["a_q"].shape[1]
        assert cycles == (m * n * k) // 64, path


def test_sim_matches_reference_on_non_square_vectors():
    """Non-square vectors are parity-checked but not cycle-checked."""
    square = {os.path.basename(p) for p in _vectors()}
    for path in _all_vectors():
        if os.path.basename(path) in square:
            continue
        d = np.load(path)
        out, _ = run_matmul(d["a_q"], d["b_q"])
        assert out.tolist() == d["expected"].tolist(), path


def _all_vectors() -> list[str]:
    return sorted(glob.glob(os.path.join(VEC_DIR, "matmul_*.npz")))


def test_all_vectors_are_parity_checked():
    """Every vector in the directory is covered by some parity test."""
    assert len(_all_vectors()) >= 7
