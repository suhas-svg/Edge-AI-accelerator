"""Developer SDK (spec section 20). Backed by the simulator until FPGA exists."""
from __future__ import annotations

import struct
import time

import numpy as np

from compiler.binary import ModelPack, read_model
from python.edge_npu.reference import bias_add, max_pool, relu, requantize
from simulator.hardware_model import MACS_PER_CYCLE, run_conv2d, run_matmul


class Model:
    def __init__(self, path: str):
        self.path = path
        try:
            self.pack: ModelPack = read_model(path)
        except (ValueError, OSError) as e:
            raise ValueError(f"cannot load model {path!r}: {e}")
        self._chain = self._validate_chain(list(self.pack.graph.nodes))
        self._cycles = 0
        self._macs = 0

    @staticmethod
    def _validate_chain(nodes: list) -> list:
        ops = [n.op for n in nodes]
        if not ops or ops[0] not in ("matmul", "conv2d"):
            raise ValueError(f"SDK chain must start with matmul or conv2d, got {ops}")
        for op in ops[1:]:
            if op not in ("relu", "bias_add", "max_pool", "requantize"):
                raise ValueError(f"SDK chain supports matmul/conv2d→relu→bias_add→max_pool→requantize, got {ops}")
        prev = nodes[0].output
        for node in nodes[1:]:
            if node.inputs[0] != prev:
                raise ValueError(
                    f"node {node.op} reads {node.inputs[0]!r}, chain holds {prev!r}"
                )
            prev = node.output
        return ops

    def predict(self, x: np.ndarray) -> np.ndarray:
        nodes = self.pack.graph.nodes
        first = nodes[0]
        if first.inputs[1] not in self.pack.weights:
            raise ValueError(f"model missing weights for {first.inputs[1]!r}")
        w = self.pack.weights[first.inputs[1]]
        if first.op == "matmul":
            a = np.ascontiguousarray(x, dtype=np.int8)
            m, k = a.shape
            kb, n = w.shape
            if k != kb:
                raise ValueError(f"input inner dim {k} vs weights {kb}")
            matmuls = [c for c in self.pack.cmds if c.op == "MATMUL"]
            if len(matmuls) != 1 or (matmuls[0].m, matmuls[0].n, matmuls[0].k) != (m, n, k):
                raise ValueError("command stream MATMUL does not match graph shapes")
            loads = [c for c in self.pack.cmds if c.op == "LOAD"]
            if [c.size for c in loads] != [a.nbytes, w.nbytes]:
                raise ValueError("command stream LOAD sizes do not match tensor bytes")
            buf, cycles = run_matmul(a, w)
            macs = m * n * k
        else:
            a = np.ascontiguousarray(x, dtype=np.int8)
            c, h, wd = a.shape
            k, kc, kh, kw = w.shape
            if kc != c:
                raise ValueError(f"input channels {c} vs weights {kc}")
            oh, ow = h - kh + 1, wd - kw + 1
            convs = [c_ for c_ in self.pack.cmds if c_.op == "CONV2D"]
            if len(convs) != 1 or (convs[0].m, convs[0].n, convs[0].k) != (k, oh, ow):
                raise ValueError("command stream CONV2D does not match graph shapes")
            loads = [c_ for c_ in self.pack.cmds if c_.op == "LOAD"]
            if [c_.size for c_ in loads] != [a.nbytes, w.nbytes]:
                raise ValueError("command stream LOAD sizes do not match tensor bytes")
            buf, cycles = run_conv2d(a, w)
            macs = k * oh * ow * c * kh * kw
        prev = first.output
        # Elementwise/pool cycles are not modeled yet; the MAC count stands.
        ew = [c_ for c_ in self.pack.cmds
              if c_.op in ("RELU", "BIAS_ADD", "MAX_POOL", "REQUANTIZE")]
        ei = 0
        for node in nodes[1:]:
            if node.inputs[0] != prev:
                raise ValueError(
                    f"node {node.op} reads {node.inputs[0]!r}, chain holds {prev!r}"
                )
            if ei >= len(ew) or ew[ei].op != node.op.upper():
                raise ValueError(f"command stream missing {node.op} for chain step")
            if node.op != "max_pool" and node.op != "requantize" and ew[ei].size != buf.nbytes:
                raise ValueError(
                    f"command stream {ew[ei].op} size does not match buffer bytes"
                )
            ei += 1
            if node.op == "relu":
                buf = relu(buf)
            elif node.op == "bias_add":
                if node.inputs[1] not in self.pack.weights:
                    raise ValueError(f"model missing weights for {node.inputs[1]!r}")
                buf = bias_add(buf, self.pack.weights[node.inputs[1]])
            elif node.op == "max_pool":
                buf = max_pool(buf, ew[ei - 1].m, ew[ei - 1].n)
            else:
                (scale,) = struct.unpack("<f", struct.pack("<I", ew[ei - 1].reserved))
                buf = requantize(buf, scale, ew[ei - 1].m - 128)
            if ew[ei - 1].size != buf.nbytes and node.op in ("max_pool", "requantize"):
                raise ValueError(f"command stream {ew[ei-1].op} size does not match output bytes")
            prev = node.output
        if ei != len(ew):
            raise ValueError("command stream has extra elementwise commands")
        stores = [c_ for c_ in self.pack.cmds if c_.op == "STORE"]
        if not stores or stores[-1].size != buf.nbytes:
            raise ValueError("command stream final STORE size does not match output bytes")
        self._cycles = cycles
        self._macs = macs
        return buf

    def get_stats(self) -> dict:
        util = self._macs / (self._cycles * MACS_PER_CYCLE) if self._cycles else 0.0
        return {"cycles": self._cycles, "mac_utilization": util,
                "memory_bandwidth_gbs": 0.0}


class Device:
    def load_model(self, path: str) -> Model:
        return Model(path)

    def info(self) -> dict:
        return {"device": "edge-npu-sim", "precision": "int8", "mac_array": "8x8"}

    def get_stats(self) -> dict:
        return {"cycles": 0, "mac_utilization": 0.0, "memory_bandwidth_gbs": 0.0}

    def benchmark(self, model: Model, x: np.ndarray, repeats: int = 5) -> dict:
        for _ in range(2):
            model.predict(x)
        samples = []
        for _ in range(repeats):
            t = time.perf_counter()
            model.predict(x)
            samples.append((time.perf_counter() - t) * 1000)
        return {"median_ms": float(np.median(samples)), "repeats": repeats}
