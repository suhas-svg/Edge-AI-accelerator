"""Harness contract. Each test fails for a real defect in the verifier."""
import os
import numpy as np
import pytest
from tools.verify_rtl import compare_vector, main


def _mk(tmp_path, expected, rtl_out, name="matmul_2x2x2.npz"):
    path = str(tmp_path / name)
    np.savez(path, a_q=np.ones((2, 2), dtype=np.int8),
             b_q=np.ones((2, 2), dtype=np.int8),
             expected=expected, rtl_out=rtl_out)
    return path


def test_matching_vector_passes(tmp_path):
    exp = np.full((2, 2), 8, dtype=np.int32)
    assert compare_vector(_mk(tmp_path, exp, exp.copy())) == []


def test_missing_rtl_out_is_a_failure(tmp_path):
    path = str(tmp_path / "matmul_2x2x2.npz")
    np.savez(path, a_q=np.ones((2, 2), dtype=np.int8),
             b_q=np.ones((2, 2), dtype=np.int8),
             expected=np.full((2, 2), 8, dtype=np.int32))
    msgs = compare_vector(path)
    assert len(msgs) == 1
    assert "MISSING" in msgs[0] and "matmul_2x2x2.npz" in msgs[0]


def test_wrong_value_reports_first_index(tmp_path):
    exp = np.full((2, 2), 8, dtype=np.int32)
    rtl = exp.copy()
    rtl[1, 1] = 9
    msgs = compare_vector(_mk(tmp_path, exp, rtl))
    assert len(msgs) == 1
    assert "(1, 1)" in msgs[0] and "8" in msgs[0] and "9" in msgs[0]


def test_shape_mismatch_reported(tmp_path):
    exp = np.full((2, 2), 8, dtype=np.int32)
    msgs = compare_vector(_mk(tmp_path, exp, np.full((1, 4), 8, dtype=np.int32)))
    assert len(msgs) == 1
    assert "(2, 2)" in msgs[0] and "(1, 4)" in msgs[0]


def test_main_exits_nonzero_on_mismatch(tmp_path):
    exp = np.full((2, 2), 8, dtype=np.int32)
    rtl = exp.copy()
    rtl[0, 0] = 7
    _mk(tmp_path, exp, rtl)
    assert main(str(tmp_path)) == 1


def test_main_exits_zero_when_all_match(tmp_path):
    exp = np.full((2, 2), 8, dtype=np.int32)
    _mk(tmp_path, exp, exp.copy())
    assert main(str(tmp_path)) == 0


def test_self_check_catches_deliberate_corruption(tmp_path):
    """The harness must fail on a corrupted copy of a real golden vector."""
    import subprocess, sys
    repo = os.path.join(os.path.dirname(__file__), "..")
    vdir = os.path.join(repo, "tests", "vectors")
    tmp = str(tmp_path)
    for f in os.listdir(vdir):
        if f.endswith(".npz"):
            d = dict(np.load(os.path.join(vdir, f)))
            d["rtl_out"] = d["expected"].copy()
            np.savez(os.path.join(tmp, f), **d)
    target = os.path.join(tmp, "matmul_8x8x8.npz")
    d = dict(np.load(target))
    d["rtl_out"] = d["rtl_out"].copy()
    d["rtl_out"][0, 0] = d["rtl_out"][0, 0] + 1
    np.savez(target, **d)
    r = subprocess.run([sys.executable, os.path.join(repo, "tools", "verify_rtl.py"), tmp],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "matmul_8x8x8.npz" in r.stdout
