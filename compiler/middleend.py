"""Compiler middle-end: schedule, optimize, validate, lower."""
from __future__ import annotations

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
