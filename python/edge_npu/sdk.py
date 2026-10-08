"""Developer SDK stub (spec section 20). Backed by the simulator until FPGA exists."""
from __future__ import annotations
import numpy as np
from .reference import matmul_int8


class Model:
    def __init__(self, path: str):
        self.path = path

    def predict(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return matmul_int8(x, w)


class Device:
    def load_model(self, path: str) -> Model:
        return Model(path)

    def info(self) -> dict:
        return {"device": "edge-npu-sim", "precision": "int8", "mac_array": "8x8"}

    def get_stats(self) -> dict:
        return {"cycles": 0, "mac_utilization": 0.0, "memory_bandwidth_gbs": 0.0}
