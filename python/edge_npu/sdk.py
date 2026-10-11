"""Developer SDK (spec section 20). Backed by the simulator until FPGA exists."""
from __future__ import annotations

import struct
import time

import numpy as np

from compiler.binary import ModelPack, read_model
from compiler.middleend import fusable
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
            if op not in ("relu", "bias_add", "max_pool", "requantize", "matmul", "conv2d"):
                raise ValueError(
                    f"SDK chain supports matmul/conv2d→relu→bias_add→max_pool→requantize→matmul/conv2d, got {ops}")
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
        matmuls = [c for c in self.pack.cmds if c.op == "MATMUL"]
        convs = [c for c in self.pack.cmds if c.op == "CONV2D"]
        loads = [c for c in self.pack.cmds if c.op == "LOAD"]
        ew = [c_ for c_ in self.pack.cmds
              if c_.op in ("RELU", "BIAS_ADD", "MAX_POOL", "REQUANTIZE")]
        fused = fusable(list(nodes))
        mi, ci, li, ei = 0, 0, 0, 0
        live: dict[str, np.ndarray] = {}
        lshape: dict[str, tuple[int, ...]] = {}
        cycles, macs = 0, 0

        def take_load(name: str, nbytes: int) -> None:
            nonlocal li
            if li >= len(loads):
                raise ValueError(f"command stream missing LOAD for {name!r}")
            if loads[li].size != nbytes:
                raise ValueError("command stream LOAD sizes do not match tensor bytes")
            li += 1

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
            if mi >= len(matmuls):
                raise ValueError("command stream MATMUL does not match graph shapes")
            mm = matmuls[mi]
            mi += 1
            a = _pad_to((mm.m, mm.k), a)
            w = _pad_to((mm.k, mm.n), w)
            take_load(first.inputs[0], a.nbytes)
            take_load(first.inputs[1], w.nbytes)
            if (mm.m, mm.n, mm.k) != (a.shape[0], w.shape[1], a.shape[1]):
                raise ValueError("command stream MATMUL does not match graph shapes")
            buf, cc = run_matmul(a, w)
            cycles += cc
            macs += mm.m * mm.n * mm.k
            live[first.output] = buf
            lshape[first.output] = (m, n)
        else:
            a = np.ascontiguousarray(x, dtype=np.int8)
            c, h, wd = a.shape
            k, kc, kh, kw = w.shape
            if kc != c:
                raise ValueError(f"input channels {c} vs weights {kc}")
            if ci >= len(convs):
                raise ValueError("command stream CONV2D does not match graph shapes")
            cv = convs[ci]
            ci += 1
            kk, ohp, owp = cv.m, cv.n, cv.k
            if li + 1 >= len(loads):
                raise ValueError(f"command stream missing LOAD for {first.inputs[1]!r}")
            cp = loads[li + 1].size // (kk * kh * kw)
            a = _pad_to((cp, ohp + kh - 1, owp + kw - 1), a)
            w = _pad_to((kk, cp, kh, kw), w)
            take_load(first.inputs[0], a.nbytes)
            take_load(first.inputs[1], w.nbytes)
            if kk < k or cp < kc:
                raise ValueError("command stream CONV2D does not match graph shapes")
            buf, cc = run_conv2d(a, w)
            cycles += cc
            macs += kk * ohp * owp * cp * kh * kw
            live[first.output] = buf
            lshape[first.output] = (k, h - kh + 1, wd - kw + 1)
        prev = first.output
        buf = live[prev]
        # Elementwise/pool cycles are not modeled yet; the MAC count stands.
        for node in nodes[1:]:
            if node.inputs[0] != prev:
                raise ValueError(
                    f"node {node.op} reads {node.inputs[0]!r}, chain holds {prev!r}"
                )
            if node.op == "matmul":
                if mi >= len(matmuls):
                    raise ValueError("command stream MATMUL does not match graph shapes")
                mm = matmuls[mi]
                mi += 1
                a_buf = live[node.inputs[0]]
                wgt = node.inputs[1]
                if wgt in live:
                    w_buf = live[wgt]
                    lw = lshape[wgt]
                else:
                    if wgt not in self.pack.weights:
                        raise ValueError(f"model missing weights for {wgt!r}")
                    wlog = self.pack.weights[wgt]
                    w_buf = _pad_to((mm.k, mm.n), wlog)
                    take_load(wgt, w_buf.nbytes)
                    lw = wlog.shape
                if (mm.m, mm.n, mm.k) != (a_buf.shape[0], w_buf.shape[1], a_buf.shape[1]):
                    raise ValueError("command stream MATMUL does not match graph shapes")
                buf, cc = run_matmul(a_buf, w_buf)
                cycles += cc
                macs += mm.m * mm.n * mm.k
                live[node.output] = buf
                lshape[node.output] = (lshape[node.inputs[0]][0], lw[1])
            elif node.op == "conv2d":
                if ci >= len(convs):
                    raise ValueError("command stream CONV2D does not match graph shapes")
                cv = convs[ci]
                ci += 1
                a_buf = live[node.inputs[0]]
                wgt = node.inputs[1]
                if wgt in live:
                    w_buf = live[wgt]
                    lw = lshape[wgt]
                else:
                    if wgt not in self.pack.weights:
                        raise ValueError(f"model missing weights for {wgt!r}")
                    wlog = self.pack.weights[wgt]
                    lw = wlog.shape
                    kk, ohp, owp = cv.m, cv.n, cv.k
                    if li >= len(loads):
                        raise ValueError(f"command stream missing LOAD for {wgt!r}")
                    cp = loads[li].size // (kk * lw[2] * lw[3])
                    w_buf = _pad_to((kk, cp, lw[2], lw[3]), wlog)
                    take_load(wgt, w_buf.nbytes)
                kk, ohp, owp = cv.m, cv.n, cv.k
                _, _, kh, kw = lw
                cp = w_buf.shape[1]
                if a_buf.shape != (cp, ohp + kh - 1, owp + kw - 1) or kk < lw[0] or cp < lw[1]:
                    raise ValueError("command stream CONV2D does not match graph shapes")
                buf, cc = run_conv2d(a_buf, w_buf)
                cycles += cc
                macs += kk * ohp * owp * cp * kh * kw
                live[node.output] = buf
                lh, lwd = lshape[node.inputs[0]][1], lshape[node.inputs[0]][2]
                lshape[node.output] = (lw[0], lh - kh + 1, lwd - kw + 1)
            else:
                if node.output in fused and (ei >= len(ew) or ew[ei].op != node.op.upper()):
                    cmd = None
                else:
                    if ei >= len(ew) or ew[ei].op != node.op.upper():
                        raise ValueError(f"command stream missing {node.op} for chain step")
                    if node.op != "max_pool" and node.op != "requantize" \
                            and ew[ei].size != buf.nbytes:
                        raise ValueError(
                            f"command stream {ew[ei].op} size does not match buffer bytes"
                        )
                    ei += 1
                    cmd = ew[ei - 1]
                if node.op == "relu":
                    buf = relu(buf)
                    lshape[node.output] = lshape[node.inputs[0]]
                elif node.op == "bias_add":
                    if node.inputs[1] not in self.pack.weights:
                        raise ValueError(f"model missing weights for {node.inputs[1]!r}")
                    b = self.pack.weights[node.inputs[1]]
                    if b.ndim == 1:
                        b = _pad_to((buf.shape[-1],), b)
                    else:
                        b = _pad_to(buf.shape, b)
                    buf = bias_add(buf, b)
                    lshape[node.output] = lshape[node.inputs[0]]
                elif node.op == "max_pool":
                    buf = max_pool(buf, cmd.m, cmd.n)
                    ch, hh, ww = lshape[node.inputs[0]]
                    lshape[node.output] = (ch, (hh - cmd.m) // cmd.n + 1,
                                           (ww - cmd.m) // cmd.n + 1)
                else:
                    (scale,) = struct.unpack("<f", struct.pack("<I", cmd.reserved))
                    buf = requantize(buf, scale, cmd.m - 128)
                    lshape[node.output] = lshape[node.inputs[0]]
                if cmd is not None and cmd.size != buf.nbytes \
                        and node.op in ("max_pool", "requantize"):
                    raise ValueError(
                        f"command stream {cmd.op} size does not match output bytes")
                live[node.output] = buf
            prev = node.output
        if ei != len(ew):
            raise ValueError("command stream has extra elementwise commands")
        if li != len(loads):
            raise ValueError("command stream has extra LOADs")
        stores = [c_ for c_ in self.pack.cmds if c_.op == "STORE"]
        if not stores or stores[-1].size != buf.nbytes:
            raise ValueError("command stream final STORE size does not match output bytes")
        logical = lshape[nodes[-1].output]
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
