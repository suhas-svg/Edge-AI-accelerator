"""Compiler middle-end: schedule, optimize, validate, lower."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from compiler.codegen import (
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
