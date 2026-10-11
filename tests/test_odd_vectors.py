"""Odd-shape golden vectors: end-to-end through compile, pack, and predict.

Tile-aligned goldens live in tests/vectors/ for the RTL harness. Odd dims
never reach hardware unpadded, so these live in tests/vectors_odd/ with the
same keys and naming, for software-side verification only.
"""
import glob
import os

import numpy as np

from compiler.binary import write_model
from compiler.graph import Graph, Node
from compiler.middleend import lower_graph
from python.edge_npu.sdk import Device
from python.edge_npu.tensor import TensorSpec
from simulator.hardware_model import run_matmul

ODD_DIR = os.path.join(os.path.dirname(__file__), "vectors_odd")


def _pad8(d: int) -> int:
    return (d + 7) // 8 * 8


def _odd_vectors() -> list[str]:
    return sorted(glob.glob(os.path.join(ODD_DIR, "matmul_*.npz")))


def test_odd_sizes_are_triples():
    from tools import gen_vectors
    assert all(len(s) == 3 for s in gen_vectors.ODD_SIZES)
    assert all(any(d % 8 for d in s) for s in gen_vectors.ODD_SIZES)


def test_odd_vectors_present():
    from tools import gen_vectors
    assert len(_odd_vectors()) == len(gen_vectors.ODD_SIZES) >= 3


def test_odd_generation_is_deterministic(tmp_path):
    import hashlib
    import subprocess
    import sys
    repo = os.path.join(os.path.dirname(__file__), "..")

    def digest() -> dict:
        out = {}
        for f in sorted(os.listdir(ODD_DIR)):
            if f.endswith(".npz"):
                with open(os.path.join(ODD_DIR, f), "rb") as fh:
                    out[f] = hashlib.sha256(fh.read()).hexdigest()
        return out

    first = digest()
    subprocess.run([sys.executable, os.path.join(repo, "tools", "gen_vectors.py")],
                   check=True, capture_output=True)
    assert digest() == first


def test_odd_vectors_end_to_end(tmp_path):
    for path in _odd_vectors():
        d = np.load(path)
        a_q, b_q, expected = d["a_q"], d["b_q"], d["expected"]
        m, k = a_q.shape
        kb, n = b_q.shape
        assert expected.shape == (m, n)
        sim_out, _ = run_matmul(a_q, b_q)
        assert sim_out.tolist() == expected.tolist(), path
        node = Node(op="matmul", inputs=("a", "w"), output="c")
        specs = {"a": TensorSpec(name="a", dtype="int8", shape=(m, k)),
                 "w": TensorSpec(name="w", dtype="int8", shape=(kb, n)),
                 "c": TensorSpec(name="c", dtype="int32", shape=(m, n))}
        lg = lower_graph(Graph(nodes=(node,)), specs, {"w": b_q}, {})
        bin_path = os.path.join(str(tmp_path),
                                os.path.basename(path) + ".bin")
        write_model(bin_path, lg.graph, lg.weights, lg.cmds)
        model = Device().load_model(bin_path)
        out = model.predict(a_q)
        assert out.shape == expected.shape, path
        np.testing.assert_array_equal(out, expected)
        assert model.get_stats()["cycles"] == \
            (_pad8(m) * _pad8(k) * _pad8(n)) // 64, path


def _odd_conv_vectors() -> list[str]:
    return sorted(glob.glob(os.path.join(ODD_DIR, "conv_*.npz")))


def _odd_pool_vectors() -> list[str]:
    return sorted(glob.glob(os.path.join(ODD_DIR, "pool_*.npz")))


def test_odd_conv_pool_sizes_present():
    from tools import gen_vectors
    assert len(_odd_conv_vectors()) == len(gen_vectors.CONV_ODD_SIZES) >= 3
    assert len(_odd_pool_vectors()) == len(gen_vectors.POOL_ODD_SIZES) >= 3


def test_odd_conv_vectors_end_to_end(tmp_path):
    from python.edge_npu.reference import conv2d_int8
    for path in _odd_conv_vectors():
        d = np.load(path)
        x_q, w_q, expected = d["x_q"], d["w_q"], d["expected"]
        assert expected.shape[1:] == (x_q.shape[1] - w_q.shape[2] + 1,
                                      x_q.shape[2] - w_q.shape[3] + 1)
        assert conv2d_int8(x_q, w_q).tolist() == expected.tolist(), path
        c, h, wd = x_q.shape
        k, _, kh, _ = w_q.shape
        oh, ow = expected.shape[1], expected.shape[2]
        node = Node(op="conv2d", inputs=("x", "w"), output="y")
        specs = {"x": TensorSpec(name="x", dtype="int8", shape=(c, h, wd)),
                 "w": TensorSpec(name="w", dtype="int8",
                                 shape=(k, c, kh, w_q.shape[3])),
                 "y": TensorSpec(name="y", dtype="int32", shape=(k, oh, ow))}
        lg = lower_graph(Graph(nodes=(node,)), specs, {"w": w_q}, {})
        bin_path = os.path.join(str(tmp_path),
                                os.path.basename(path) + ".bin")
        write_model(bin_path, lg.graph, lg.weights, lg.cmds)
        out = Device().load_model(bin_path).predict(x_q)
        assert out.shape == expected.shape, path
        np.testing.assert_array_equal(out, expected)


def test_odd_pool_vectors_match_reference():
    from python.edge_npu.reference import max_pool
    for path in _odd_pool_vectors():
        d = np.load(path)
        base = os.path.basename(path)[:-len(".npz")]
        _, dims, win, stride = base.split("_")
        size, st = int(win[1:]), int(stride[1:])
        assert max_pool(d["x"], size, st).tolist() == d["expected"].tolist(), path
