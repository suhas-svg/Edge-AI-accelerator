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


def test_sim_matches_reference_on_all_vectors():
    for path in _vectors():
        d = np.load(path)
        out, cycles = run_matmul(d["a_q"], d["b_q"])
        assert out.tolist() == d["expected"].tolist(), path
        m, n = d["expected"].shape
        k = d["a_q"].shape[1]
        assert cycles == (m * n * k) // 64, path
