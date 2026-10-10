"""Graph types. Parse once at the edge, trust the interior."""
from __future__ import annotations

from dataclasses import dataclass

VALID_OPS: tuple[str, ...] = (
    "matmul", "relu", "bias_add", "conv2d", "max_pool", "requantize", "load", "store",
)


@dataclass(frozen=True)
class Node:
    op: str
    inputs: tuple[str, ...]
    output: str

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"unknown op {self.op!r}")
        if not self.output:
            raise ValueError("node needs an output name")


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for n in self.nodes:
            if n.output in seen:
                raise ValueError(f"duplicate output {n.output!r}")
            seen.add(n.output)

    def order(self) -> list[Node]:
        return list(self.nodes)
