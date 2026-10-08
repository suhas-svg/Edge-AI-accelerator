"""Vector generation contract. Each fails for a real generation defect."""
import os
import numpy as np
from tools import gen_vectors


def test_sizes_are_triples():
    assert all(len(s) == 3 for s in gen_vectors.SIZES)


def test_generation_is_byte_identical():
    import hashlib, subprocess, sys, os
    repo = os.path.join(os.path.dirname(__file__), "..")

    def digest() -> dict:
        out = {}
        d = os.path.join(repo, "tests", "vectors")
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz"):
                with open(os.path.join(d, f), "rb") as fh:
                    out[f] = hashlib.sha256(fh.read()).hexdigest()
        return out

    first = digest()
    subprocess.run([sys.executable, os.path.join(repo, "tools", "gen_vectors.py")],
                   check=True, capture_output=True)
    assert digest() == first


def test_overflow_vector_expected_value():
    d = np.load(os.path.join("tests", "vectors", "matmul_overflow.npz"))
    assert d["a_q"].shape == (64, 64)
    assert d["a_q"].min() == -128 and d["a_q"].max() == -128
    assert d["b_q"].min() == 127 and d["b_q"].max() == 127
    assert d["expected"].tolist() == [[-1040384] * 64] * 64
