"""Compiler middle-end: schedule, optimize, validate, lower."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from compiler.codegen import (
    TILE,
    emit_matmul,
    lower_bias_add,
    lower_conv2d,
    lower_max_pool,
    lower_relu,
    lower_requantize,
)
from compiler.graph import Graph, Node
from compiler.memory import MemoryConfig, MemoryPlan, plan_memory
from python.edge_npu.commands import Command
from python.edge_npu.tensor import TensorSpec

_DTYPE_BYTES = {"fp32": 4, "int8": 1, "int32": 4}


def schedule(graph: Graph, specs: dict[str, TensorSpec]) -> list[Node]:
    """Deterministic topological order; ties break by original tuple index."""
    nodes = list(graph.nodes)
    n = len(nodes)
    producer: dict[str, int] = {}
    for i, nd in enumerate(nodes):
        producer[nd.output] = i
    for nd in nodes:
        if nd.output not in specs:
            raise ValueError(f"tensor {nd.output!r} not in specs")
    deps: list[set[int]] = []
    consumers: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, nd in enumerate(nodes):
        seen: set[int] = set()
        for inp in nd.inputs:
            if inp in producer:
                seen.add(producer[inp])
            elif inp not in specs:
                raise ValueError(f"tensor {inp!r} not in specs")
        deps.append(seen)
        for p in seen:
            consumers[p].append(i)
    unmet = [len(d) for d in deps]
    ready = sorted(i for i, c in enumerate(unmet) if c == 0)
    order: list[Node] = []
    emitted = [False] * n
    while ready:
        i = ready.pop(0)
        emitted[i] = True
        order.append(nodes[i])
        for j in consumers[i]:
            unmet[j] -= 1
            if unmet[j] == 0:
                ready.append(j)
                ready.sort()
    if len(order) != n:
        remaining = sorted(nodes[i].output for i, done in enumerate(emitted) if not done)
        raise ValueError(f"graph has a cycle involving {remaining!r}")
    return order


def optimize(graph: Graph,
             weights: dict[str, np.ndarray]) -> tuple[Graph, dict[str, np.ndarray]]:
    """Collapse relu->relu chains, prune unreferenced weights. Fixed point."""
    nodes = list(graph.nodes)
    changed = True
    while changed:
        changed = False
        by_output = {n.output: n for n in nodes}
        for b in nodes:
            if b.op != "relu" or len(b.inputs) != 1:
                continue
            a = by_output.get(b.inputs[0])
            if a is None or a.op != "relu":
                continue
            nodes = [n for n in nodes if n.output != b.output]
            nodes = [Node(op=n.op,
                          inputs=tuple(a.output if i == b.output else i for i in n.inputs),
                          output=n.output) for n in nodes]
            changed = True
            break
    referenced: set[str] = set()
    for n in nodes:
        referenced.update(n.inputs)
    pruned = {k: v for k, v in weights.items() if k in referenced}
    return Graph(nodes=tuple(nodes)), pruned


_LOWERED_OPS = ("matmul", "conv2d", "relu", "bias_add", "max_pool", "requantize")


def validate(order: list[Node]) -> None:
    """Chain legality for lowering: no fan-out, one sink, supported ops."""
    produced = {n.output for n in order}
    consumers: dict[str, int] = {name: 0 for name in produced}
    for node in order:
        for inp in node.inputs:
            if inp in consumers:
                consumers[inp] += 1
    for name, count in consumers.items():
        if count > 1:
            raise ValueError(f"tensor {name!r} feeds {count} nodes (fan-out)")
    sinks = [n.output for n in order if consumers[n.output] == 0]
    if len(sinks) != 1:
        raise ValueError(f"graph must have a single output; found {len(sinks)}")
    for node in order:
        if node.op not in _LOWERED_OPS:
            raise ValueError(f"no lowering for op {node.op!r}")


@dataclass
class LoweredGraph:
    graph: Graph
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    cmds: list[Command] = field(default_factory=list)
    plan: MemoryPlan = field(default_factory=MemoryPlan)


def _tensor_bytes(spec: TensorSpec) -> int:
    return math.prod(spec.shape) * _DTYPE_BYTES[spec.dtype]


def lower_graph(graph: Graph, specs: dict[str, TensorSpec],
                weights: dict[str, np.ndarray], attrs: dict[str, dict],
                *, memory: MemoryConfig | None = None) -> LoweredGraph:
    """Schedule, optimize, validate, plan, and lower to a command stream."""
    config = memory if memory is not None else MemoryConfig()
    opt_graph, opt_weights = optimize(graph, weights)
    order = schedule(opt_graph, specs)
    specs = legalize(order, specs, attrs)
    validate(order)
    aliases = {n.output: n.inputs[0] for n in order if n.op in ("relu", "bias_add")}
    plan = plan_memory(order, specs, aliases, config)
    addr = {name: slot.address for name, slot in plan.slots.items()}
    cmds: list[Command] = []
    for node in order:
        out_addr = addr[node.output]
        if node.op == "matmul":
            cmds += emit_matmul(node, specs, addr[node.inputs[0]],
                                addr[node.inputs[1]], out_addr)
        elif node.op == "conv2d":
            x, w = node.inputs[0], node.inputs[1]
            cmds += [Command(op="LOAD", address=addr[x], size=_tensor_bytes(specs[x])),
                     Command(op="LOAD", address=addr[w], size=_tensor_bytes(specs[w]))]
            cmds += lower_conv2d(node, specs, base=out_addr)
        elif node.op == "relu":
            cmds += lower_relu(node, specs, base=out_addr)
        elif node.op == "bias_add":
            cmds += lower_bias_add(node, specs, base=out_addr)
        elif node.op == "max_pool":
            if node.output not in attrs:
                raise ValueError(f"pool node {node.output!r} has no attrs")
            a = attrs[node.output]
            cmds += lower_max_pool(node, specs, base=out_addr,
                                   size=a["size"], stride=a["stride"])
        elif node.op == "requantize":
            cmds += lower_requantize(node, specs, base=out_addr,
                                     scale=specs[node.output].scale)
        else:  # pragma: no cover — validate rejects these above
            raise ValueError(f"no lowering for op {node.op!r}")
    consumed = {i for n in order for i in n.inputs}
    sink = next(n.output for n in order if n.output not in consumed)
    sink_size = _tensor_bytes(specs[sink])
    if not (cmds and cmds[-1].op == "STORE"
            and cmds[-1].address == addr[sink] and cmds[-1].size == sink_size):
        cmds.append(Command(op="STORE", address=addr[sink], size=sink_size))
    return LoweredGraph(graph=Graph(nodes=tuple(order)), weights=opt_weights,
                        cmds=cmds, plan=plan)


def _pad8(dim: int) -> int:
    return (dim + 7) // 8 * 8


def _pad_elem(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Pad every dim to a tile multiple; extend the last until numel % 64 == 0."""
    out = tuple(_pad8(d) for d in shape)
    while math.prod(out) % (TILE * TILE) != 0:
        out = out[:-1] + (out[-1] + TILE,)
    return out


def legalize(order: list[Node], specs: dict[str, TensorSpec],
             attrs: dict[str, dict]) -> dict[str, TensorSpec]:
    """Verify logical shapes, rewrite specs to padded tile geometry."""
    out = dict(specs)
    padded: dict[str, tuple[int, ...]] = {}

    def put(name: str, shape: tuple[int, ...]) -> None:
        s = specs[name]
        padded[name] = shape
        out[name] = TensorSpec(name=s.name, dtype=s.dtype, shape=shape,
                               scale=s.scale, zero_point=s.zero_point)

    def pad_in(name: str) -> tuple[int, ...]:
        if name in padded:
            return padded[name]
        shape = specs[name].shape
        p = tuple(_pad8(d) for d in shape) if len(shape) >= 2 else shape
        put(name, p)
        return p

    for node in order:
        if node.op == "matmul":
            a_s, w_s, c_s = specs[node.inputs[0]], specs[node.inputs[1]], specs[node.output]
            for t in (a_s, w_s, c_s):
                if len(t.shape) != 2:
                    raise ValueError(f"tensor {t.name!r} must be rank 2, got shape {t.shape}")
            m, ka = a_s.shape
            kb, n = w_s.shape
            if ka != kb:
                raise ValueError(f"inner dim mismatch {ka} vs {kb}")
            if c_s.shape != (m, n):
                raise ValueError(f"output shape {c_s.shape} != matmul result ({m}, {n})")
            pa, pw = pad_in(node.inputs[0]), pad_in(node.inputs[1])
            kp = max(pa[1], pw[0])
            pa, pw = (pa[0], kp), (kp, pw[1])
            put(node.inputs[0], pa)
            put(node.inputs[1], pw)
            put(node.output, (pa[0], pw[1]))
        elif node.op == "conv2d":
            x_s, w_s, y_s = specs[node.inputs[0]], specs[node.inputs[1]], specs[node.output]
            if len(x_s.shape) != 3:
                raise ValueError(f"conv input must be (C,H,W), got {x_s.shape}")
            if len(w_s.shape) != 4:
                raise ValueError(f"conv weight must be (K,C,KH,KW), got {w_s.shape}")
            if len(y_s.shape) != 3:
                raise ValueError(f"conv output must be (K,OH,OW), got {y_s.shape}")
            c, h, wd = x_s.shape
            k, kc, kh, kw = w_s.shape
            if kc != c:
                raise ValueError(f"conv channel mismatch weight {kc} vs input {c}")
            if kh != kw:
                raise ValueError(f"conv kernel must be square, got ({kh}, {kw})")
            oh, ow = h - kh + 1, wd - kw + 1
            if y_s.shape != (k, oh, ow):
                raise ValueError(f"conv output shape {y_s.shape} != ({k}, {oh}, {ow})")
            cp = _pad8(max(c, kc))
            ohp, owp = _pad8(oh), _pad8(ow)
            put(node.inputs[1], (_pad8(k), cp, kh, kw))
            if node.inputs[0] in padded:
                px = padded[node.inputs[0]]
            else:
                px = (cp, ohp + kh - 1, owp + kw - 1)
                put(node.inputs[0], px)
            put(node.output, (_pad8(k), ohp, owp))
        elif node.op in ("relu", "requantize"):
            if len(node.inputs) != 1:
                raise ValueError(f"{node.op} needs 1 inputs, got {len(node.inputs)}")
            x_s, y_s = specs[node.inputs[0]], specs[node.output]
            if node.op == "relu" and x_s.shape != y_s.shape:
                raise ValueError(f"relu input shape {x_s.shape} != output {y_s.shape}")
            if node.op == "requantize" and x_s.shape != y_s.shape:
                raise ValueError(f"requantize shape {x_s.shape} != output {y_s.shape}")
            if len(y_s.shape) not in (2, 3):
                raise ValueError(f"tensor {y_s.name!r} must be rank 2 or 3, got shape {y_s.shape}")
            p = _pad_elem(pad_in(node.inputs[0]))
            put(node.inputs[0], p)
            put(node.output, p)
        elif node.op == "bias_add":
            if len(node.inputs) != 2:
                raise ValueError(f"{node.op} needs 2 inputs, got {len(node.inputs)}")
            data_s, bias_s, out_s = specs[node.inputs[0]], specs[node.inputs[1]], specs[node.output]
            if data_s.shape != out_s.shape:
                raise ValueError(f"bias_add input shape {data_s.shape} != output {out_s.shape}")
            if bias_s.shape != (out_s.shape[-1],) and bias_s.shape != out_s.shape:
                raise ValueError(
                    f"bias shape {bias_s.shape} broadcasts neither to last dim {out_s.shape[-1]} "
                    f"nor to {out_s.shape}")
            if len(out_s.shape) not in (2, 3):
                raise ValueError(f"tensor {out_s.name!r} must be rank 2 or 3, got shape {out_s.shape}")
            pd = _pad_elem(pad_in(node.inputs[0]))
            put(node.inputs[0], pd)
            put(node.output, pd)
            if len(bias_s.shape) == 1:
                put(node.inputs[1], (pd[-1],))
            else:
                put(node.inputs[1], pd)
        elif node.op == "max_pool":
            if len(node.inputs) != 1:
                raise ValueError(f"{node.op} needs 1 inputs, got {len(node.inputs)}")
            if node.output not in attrs:
                raise ValueError(f"pool node {node.output!r} has no attrs")
            a = attrs[node.output]
            size, stride = a["size"], a["stride"]
            if size <= 0 or stride <= 0:
                raise ValueError(f"pool size={size} stride={stride} must be positive")
            x_s, y_s = specs[node.inputs[0]], specs[node.output]
            if len(x_s.shape) != 3 or len(y_s.shape) != 3:
                raise ValueError(f"pool tensors must be (C,H,W), got {x_s.shape} → {y_s.shape}")
            c, h, wd = x_s.shape
            oh, ow = (h - size) // stride + 1, (wd - size) // stride + 1
            if oh <= 0 or ow <= 0:
                raise ValueError(f"pool window {size} larger than input ({h}, {wd})")
            if y_s.shape != (c, oh, ow):
                raise ValueError(f"pool output shape {y_s.shape} != ({c}, {oh}, {ow})")
            px = pad_in(node.inputs[0])
            put(node.output, (px[0], (px[1] - size) // stride + 1,
                              (px[2] - size) // stride + 1))
        else:  # pragma: no cover — validate rejects these in lower_graph
            raise ValueError(f"no lowering for op {node.op!r}")
    return out
