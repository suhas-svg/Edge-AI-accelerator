"""Developer SDK (spec section 20). Backed by the simulator until FPGA exists."""
from __future__ import annotations

import struct
import time

import numpy as np

from compiler.binary import ModelPack, read_model
from python.edge_npu.reference import bias_add, max_pool, relu, requantize
from simulator.hardware_model import MACS_PER_CYCLE, run_conv2d, run_matmul


def _pad_to(shape: tuple[int, ...], arr: np.ndarray) -> np.ndarray:
    """Zero-pad an array out to the padded tile geometry, per dimension."""
    shape = tuple(shape)
    if len(shape) != arr.ndim or any(s < l for s, l in zip(shape, arr.shape)):
        raise ValueError(
            f"tensor shape {arr.shape} does not fit tile geometry {shape}")
    if tuple(arr.shape) == shape:
        return np.ascontiguousarray(arr)
    out = np.zeros(shape, dtype=arr.dtype)
    out[tuple(slice(0, d) for d in arr.shape)] = arr
    return out


def _logical_sink_shape(nodes, cmds, x_shape: tuple[int, ...],
                        w_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Propagate logical (unpadded) shapes through the chain to the sink."""
    first = nodes[0]
    if first.op == "matmul":
        shape = (x_shape[0], w_shape[1])
    else:
        c, h, wd = x_shape
        k, _kc, kh, kw = w_shape
        shape = (k, h - kh + 1, wd - kw + 1)
    ew = [c_ for c_ in cmds if c_.op in ("RELU", "BIAS_ADD", "MAX_POOL", "REQUANTIZE")]
    ei = 0
    for node in nodes[1:]:
        if node.op == "max_pool":
            cmd = ew[ei]
            ch, hh, ww = shape
            shape = (ch, (hh - cmd.m) // cmd.n + 1, (ww - cmd.m) // cmd.n + 1)
        ei += 1
    return shape


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
        self._dma_bytes = sum(c.size for c in self.pack.cmds
                              if c.op in ("LOAD", "STORE"))
        self._cmd_counts: dict[str, int] = {}
        for c in self.pack.cmds:
            self._cmd_counts[c.op] = self._cmd_counts.get(c.op, 0) + 1

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
            if len(matmuls) != 1:
                raise ValueError("command stream MATMUL does not match graph shapes")
            mm = matmuls[0]
            loads = [c for c in self.pack.cmds if c.op == "LOAD"]
            x_shape, w_shape = a.shape, w.shape
            a = _pad_to((mm.m, mm.k), a)
            w = _pad_to((mm.k, mm.n), w)
            if [c.size for c in loads] != [a.nbytes, w.nbytes]:
                raise ValueError("command stream LOAD sizes do not match tensor bytes")
            if (mm.m, mm.n, mm.k) != (a.shape[0], w.shape[1], a.shape[1]):
                raise ValueError("command stream MATMUL does not match graph shapes")
            buf, cycles = run_matmul(a, w)
            macs = mm.m * mm.n * mm.k
        else:
            a = np.ascontiguousarray(x, dtype=np.int8)
            c, h, wd = a.shape
            k, kc, kh, kw = w.shape
            if kc != c:
                raise ValueError(f"input channels {c} vs weights {kc}")
            convs = [c_ for c_ in self.pack.cmds if c_.op == "CONV2D"]
            if len(convs) != 1:
                raise ValueError("command stream CONV2D does not match graph shapes")
            cv = convs[0]
            loads = [c_ for c_ in self.pack.cmds if c_.op == "LOAD"]
            x_shape, w_shape = a.shape, w.shape
            kk, ohp, owp = cv.m, cv.n, cv.k
            cp = loads[1].size // (kk * kh * kw)
            a = _pad_to((cp, ohp + kh - 1, owp + kw - 1), a)
            w = _pad_to((kk, cp, kh, kw), w)
            if [c_.size for c_ in loads] != [a.nbytes, w.nbytes]:
                raise ValueError("command stream LOAD sizes do not match tensor bytes")
            if kk < k or cp < kc:
                raise ValueError("command stream CONV2D does not match graph shapes")
            buf, cycles = run_conv2d(a, w)
            macs = kk * ohp * owp * cp * kh * kw
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
                b = self.pack.weights[node.inputs[1]]
                if b.ndim == 1:
                    b = _pad_to((buf.shape[-1],), b)
                else:
                    b = _pad_to(buf.shape, b)
                buf = bias_add(buf, b)
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
        logical = _logical_sink_shape(nodes, self.pack.cmds, x_shape, w_shape)
        buf = buf[tuple(slice(0, d) for d in logical)]
        self._cycles = cycles
        self._macs = macs
        return buf

    def get_stats(self) -> dict:
        util = self._macs / (self._cycles * MACS_PER_CYCLE) if self._cycles else 0.0
        return {"cycles": self._cycles, "mac_utilization": util,
                "memory_bandwidth_gbs": 0.0, "busy_cycles": self._cycles,
                "idle_cycles": 0, "dma_bytes": self._dma_bytes,
                "command_counts": dict(self._cmd_counts)}


class Device:
    def __init__(self) -> None:
        self._open = True

    def _require_open(self) -> None:
        if not self._open:
            raise ValueError("device is closed after reset; call open() first")

    def open(self) -> None:
        self._open = True

    def reset(self) -> None:
        self._require_open()
        self._open = False

    def load_model(self, path: str) -> Model:
        self._require_open()
        return Model(path)

    def info(self) -> dict:
        self._require_open()
        return {"device": "edge-npu-sim", "precision": "int8", "mac_array": "8x8"}

    def get_stats(self) -> dict:
        self._require_open()
        return {"cycles": 0, "mac_utilization": 0.0, "memory_bandwidth_gbs": 0.0}

    def benchmark(self, model: Model, x: np.ndarray, repeats: int = 5) -> dict:
        self._require_open()
        for _ in range(2):
            model.predict(x)
        samples = []
        for _ in range(repeats):
            t = time.perf_counter()
            model.predict(x)
            samples.append((time.perf_counter() - t) * 1000)
        return {"median_ms": float(np.median(samples)), "repeats": repeats}
