"""Middle-end memory planner: liveness slots over a scratch region."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from compiler.graph import Node
from python.edge_npu.tensor import TensorSpec

_DTYPE_BYTES = {"fp32": 4, "int8": 1, "int32": 4}


@dataclass(frozen=True)
class MemoryConfig:
    base: int = 0x1000
    size: int = 0x100000
    align: int = 64

    def __post_init__(self) -> None:
        if self.base < 0:
            raise ValueError(f"memory base must be >= 0, got {self.base}")
        if self.size <= 0:
            raise ValueError(f"memory size must be positive, got {self.size}")
        if self.align <= 0:
            raise ValueError(f"memory align must be positive, got {self.align}")
        if self.base % self.align != 0:
            raise ValueError(
                f"memory base {self.base:#x} is not a multiple of align {self.align}")


@dataclass(frozen=True)
class Slot:
    address: int
    size: int


@dataclass
class MemoryPlan:
    slots: dict[str, Slot] = field(default_factory=dict)
    peak_bytes: int = 0
    config: MemoryConfig = field(default_factory=MemoryConfig)


def _align_up(v: int, align: int) -> int:
    return (v + align - 1) // align * align


def _byte_size(spec: TensorSpec) -> int:
    return math.prod(spec.shape) * _DTYPE_BYTES[spec.dtype]


def _resolve_root(name: str, aliases: dict[str, str]) -> str:
    seen = {name}
    while name in aliases:
        name = aliases[name]
        if name in seen:
            raise ValueError(f"alias cycle involving {name!r}")
        seen.add(name)
    return name


def plan_memory(order: list[Node], specs: dict[str, TensorSpec],
                aliases: dict[str, str], config: MemoryConfig) -> MemoryPlan:
    """One slot per referenced tensor; first-fit reuse after last use."""
    first_use: dict[str, int] = {}
    last_use: dict[str, int] = {}
    for i, node in enumerate(order):
        for name in (*node.inputs, node.output):
            if name not in first_use:
                first_use[name] = i
        for name in node.inputs:
            last_use[name] = i
    for name in first_use:
        last_use.setdefault(name, len(order))

    slot_of: dict[str, Slot] = {}
    tenants: dict[str, set[str]] = {}
    spans: dict[str, tuple[int, int]] = {}
    free: list[tuple[int, int]] = []
    bump = config.base

    def alloc(name: str, size: int) -> None:
        nonlocal bump
        span = _align_up(size, config.align)
        for k, (start, end) in enumerate(free):
            if end - start >= span:
                addr = start
                if end - start == span:
                    del free[k]
                else:
                    free[k] = (start + span, end)
                slot_of[name] = Slot(address=addr, size=size)
                tenants[name] = {name}
                spans[name] = (addr, addr + span)
                return
        addr = _align_up(bump, config.align)
        bump = addr + span
        slot_of[name] = Slot(address=addr, size=size)
        tenants[name] = {name}
        spans[name] = (addr, addr + span)

    for i, node in enumerate(order):
        for inp in node.inputs:
            if first_use[inp] == i and inp != node.output and inp not in slot_of:
                alloc(inp, _byte_size(specs[inp]))
        out = node.output
        if out in aliases:
            root = _resolve_root(out, aliases)
            if root not in slot_of:
                raise ValueError(f"alias target {root!r} has no slot")
            slot_of[out] = slot_of[root]
            tenants[root].add(out)
        elif out not in slot_of:
            alloc(out, _byte_size(specs[out]))
        for name, last in last_use.items():
            if last != i or name not in slot_of:
                continue
            root = _resolve_root(name, aliases) if name in aliases else name
            live = tenants[root]
            live.discard(name)
            if not live:
                start, end = spans[root]
                free.append((start, end))
                free.sort()

    peak = 0
    for slot in slot_of.values():
        peak = max(peak, slot.address + slot.size)
    peak_bytes = peak - config.base if slot_of else 0
    if slot_of and peak > config.base + config.size:
        raise ValueError(
            f"memory plan needs {peak_bytes} bytes, budget {config.size}")
    return MemoryPlan(slots=slot_of, peak_bytes=peak_bytes, config=config)
