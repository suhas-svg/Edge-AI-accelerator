"""Developer SDK (spec section 20). Backed by the simulator until FPGA exists."""
from __future__ import annotations

import time

import numpy as np

from compiler.binary import ModelPack, read_model
from python.edge_npu.reference import bias_add, relu
from simulator.hardware_model import MACS_PER_CYCLE, run_matmul


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
        if not ops or ops[0] != "matmul":
            raise ValueError(f"SDK chain must start with matmul, got {ops}")
        for op in ops[1:]:
            if op not in ("relu", "bias_add"):
                raise ValueError(f"SDK chain supports matmul→relu→bias_add, got {ops}")
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
        mm = nodes[0]
        if mm.inputs[1] not in self.pack.weights:
            raise ValueError(f"model missing weights for {mm.inputs[1]!r}")
        w = self.pack.weights[mm.inputs[1]]
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
        prev = mm.output
        # Elementwise cycles are not modeled yet; the MAC count stands.
        ew = [c for c in self.pack.cmds if c.op in ("RELU", "BIAS_ADD")]
        ei = 0
        for node in nodes[1:]:
            if node.inputs[0] != prev:
                raise ValueError(
                    f"node {node.op} reads {node.inputs[0]!r}, chain holds {prev!r}"
                )
            if ei >= len(ew) or ew[ei].op != node.op.upper():
                raise ValueError(f"command stream missing {node.op} for chain step")
            if ew[ei].size != buf.nbytes:
                raise ValueError(
                    f"command stream {ew[ei].op} size does not match buffer bytes"
                )
            ei += 1
            if node.op == "relu":
                buf = relu(buf)
            else:
                if node.inputs[1] not in self.pack.weights:
                    raise ValueError(f"model missing weights for {node.inputs[1]!r}")
                buf = bias_add(buf, self.pack.weights[node.inputs[1]])
            prev = node.output
        if ei != len(ew):
            raise ValueError("command stream has extra elementwise commands")
        stores = [c for c in self.pack.cmds if c.op == "STORE"]
        if [c.size for c in stores] != [buf.nbytes]:
            raise ValueError("command stream STORE size does not match output bytes")
        self._cycles = cycles
        self._macs = m * n * k
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
