"""Developer SDK (spec section 20). Backed by the simulator until FPGA exists."""
from __future__ import annotations

import time

import numpy as np

from compiler.binary import ModelPack, read_model
from simulator.hardware_model import MACS_PER_CYCLE, run_matmul


class Model:
    def __init__(self, path: str):
        self.path = path
        try:
            self.pack: ModelPack = read_model(path)
        except (ValueError, OSError) as e:
            raise ValueError(f"cannot load model {path!r}: {e}")
        nodes = [n for n in self.pack.graph.nodes if n.op == "matmul"]
        if len(nodes) != 1 or len(self.pack.graph.nodes) != 1:
            raise ValueError(
                f"SDK executes a single matmul graph, got {[n.op for n in self.pack.graph.nodes]}"
            )
        self._node = nodes[0]
        self._cycles = 0
        self._macs = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        node = self._node
        if node.inputs[1] not in self.pack.weights:
            raise ValueError(f"model missing weights for {node.inputs[1]!r}")
        w = self.pack.weights[node.inputs[1]]
        a = np.ascontiguousarray(x, dtype=np.int8)
        m, k = a.shape
        kb, n = w.shape
        if k != kb:
            raise ValueError(f"input inner dim {k} vs weights {kb}")
        matmuls = [c for c in self.pack.cmds if c.op == "MATMUL"]
        if len(matmuls) != 1 or (matmuls[0].m, matmuls[0].n, matmuls[0].k) != (m, n, k):
            raise ValueError("command stream MATMUL does not match graph shapes")
        loads = [c for c in self.pack.cmds if c.op == "LOAD"]
        stores = [c for c in self.pack.cmds if c.op == "STORE"]
        if [c.size for c in loads] != [a.nbytes, w.nbytes]:
            raise ValueError("command stream LOAD sizes do not match tensor bytes")
        out, cycles = run_matmul(a, w)
        if [c.size for c in stores] != [out.nbytes]:
            raise ValueError("command stream STORE size does not match output bytes")
        self._cycles = cycles
        self._macs = m * n * k
        return out

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
