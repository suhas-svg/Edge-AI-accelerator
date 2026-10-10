"""Compiler middle-end: schedule, optimize, validate, lower."""
from __future__ import annotations

import numpy as np

from compiler.graph import Graph, Node
from python.edge_npu.tensor import TensorSpec


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
