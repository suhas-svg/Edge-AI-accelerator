# Compiler middle-end core implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the compiler middle-end core — deterministic scheduling, graph passes, liveness-based memory planning, and a `lower_graph` driver — so model.bin streams are produced by the compiler instead of hand-assembled at every call site.

**Architecture:** New modules `compiler/middleend.py` (schedule → optimize → validate → lower) and `compiler/memory.py` (liveness + first-fit planner over a scratch region) sit between quantize and codegen. `lower_graph` reuses the existing per-op lowerings with planned addresses and emits the final STORE; codegen gains an address-explicit matmul emitter used only by the driver. No wire-format changes: command v0.4 and model.bin v0.1 stay frozen, and every existing lowering keeps its signature and output.

**Tech Stack:** Python 3.12 (`.venv`), numpy, pytest; gcc checks via `scripts/check.sh`.

**Spec:** the user-confirmed middle-end design for this plan — two PRs (this is the core; tiling follows), canonicalize + prune + loud legality errors, pad-and-trim deferred — plus the in-repo contracts `docs/architecture.md`, `docs/memory-map.md`, `docs/command-format.md`, `docs/model-format.md`. No standalone middle-end spec section is reachable in-repo; rulings made while executing are provisional against those contracts.

## Global Constraints

- Environment repair is mandatory before any python/gcc run: `/c/MinGW/bin` first on `PATH` and `PYTHONPATH` unset. `bash scripts/check.sh` does both; for direct runs use `.venv/Scripts/python.exe -m pytest`.
- Contracts frozen: no edits to `docs/command-format.md` or `docs/model-format.md` in this plan.
- `Node` and `Graph` stay as they are; pool params stay in the frontend `attrs` side-table.
- Sim cycles stay computed (MACs/64); this plan does not touch the cycle model.
- Existing test files must not be modified — they pin the behavior a refactor must preserve. New tests go in new files (`tests/test_middleend.py`, `tests/test_memory.py`).
- Every task ends green on `.venv/Scripts/python.exe -m pytest tests/ -q`; the branch ends green on `bash scripts/check.sh` ("check ok").

## Review Focus

- Empty graph: `lower_graph(Graph(nodes=()))` must be a named `ValueError`, not a downstream crash. Pinned in Task 4.
- Shared weights: one weight feeding two nodes is legitimate; a node-produced tensor feeding two nodes is fan-out and must be a named error. Pinned in Task 4.
- Requantize scale is taken verbatim from the output tensor's spec scale; deriving it from upstream quantize scales is explicitly out of scope. Pinned in Task 5.
- Memory budget boundary: a plan needing exactly the budget passes; one byte less is a named error carrying both numbers. Pinned in Task 3.
- SDK compatibility: a stream produced by `lower_graph` still satisfies the SDK chain checks end-to-end (load order/sizes, per-step size vs live buffer, final STORE). Pinned in Task 5.

**Follow-up (separate PR, not this plan):** odd-dim legalization ("tiling"): middle-end pads dims to 8-multiples in planned buffers, commands stay tile-aligned, the SDK computes padded and trims to the logical shape at the model boundary.

---

### Task 1: Scheduler

**Files:**
- Create: `compiler/middleend.py`
- Test: `tests/test_middleend.py` (new file)

**Interfaces:**
- Consumes: `Graph`, `Node` from `compiler.graph`; `TensorSpec` from `python.edge_npu.tensor`.
- Produces: `schedule(graph: Graph, specs: dict[str, TensorSpec]) -> list[Node]` — deterministic topological order (ties broken by original tuple index); raises `ValueError` naming tensors for a cycle, an input missing from `specs`, or a node output missing from `specs`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from compiler.graph import Graph, Node
from compiler.middleend import schedule
from python.edge_npu.tensor import TensorSpec


def _specs(*names, dtype="int32", shape=(8, 8)):
    return {n: TensorSpec(name=n, dtype=dtype, shape=shape) for n in names}


def test_schedule_chain_preserves_order():
    matmul = Node(op="matmul", inputs=("a", "w"), output="c")
    relu = Node(op="relu", inputs=("c",), output="r")
    order = schedule(Graph(nodes=(matmul, relu)), _specs("a", "w", "c", "r"))
    assert [n.output for n in order] == ["c", "r"]


def test_schedule_reorders_reversed_tuple():
    matmul = Node(op="matmul", inputs=("a", "w"), output="c")
    relu = Node(op="relu", inputs=("c",), output="r")
    order = schedule(Graph(nodes=(relu, matmul)), _specs("a", "w", "c", "r"))
    assert [n.output for n in order] == ["c", "r"]


def test_schedule_names_cycle():
    g = Graph(nodes=(Node(op="relu", inputs=("y",), output="x"),
                     Node(op="relu", inputs=("x",), output="y")))
    with pytest.raises(ValueError, match="cycle"):
        schedule(g, _specs("x", "y"))


def test_schedule_rejects_unknown_input():
    g = Graph(nodes=(Node(op="relu", inputs=("zz",), output="r"),))
    with pytest.raises(ValueError, match="zz"):
        schedule(g, _specs("r"))


def test_schedule_rejects_output_missing_from_specs():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    with pytest.raises(ValueError, match="'c'"):
        schedule(g, _specs("a", "w"))


def test_schedule_deterministic_tie_break():
    # Tuple order is (C, A, B); A and B are both ready first, A has the lower index.
    c = Node(op="bias_add", inputs=("y", "q"), output="z")
    a = Node(op="relu", inputs=("p",), output="q")
    b = Node(op="relu", inputs=("x",), output="y")
    order = schedule(Graph(nodes=(c, a, b)), _specs("p", "q", "x", "y", "z"))
    assert [n.output for n in order] == ["q", "y", "z"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_middleend.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'compiler.middleend'`.

- [ ] **Step 3: Implement `schedule` in `compiler/middleend.py`**

Kahn's algorithm. Build `producer: dict[str, int]` (output → node index). A node is initially ready when none of its inputs have a producer. Keep ready nodes in a list and always pop the lowest original index (deterministic). When a node is emitted, decrement its consumers' unmet-producer counts; at zero the consumer joins ready. Names are validated as they are first seen: an input with no producer that is not in `specs`, or a node output not in `specs`, raises `ValueError` naming the tensor (`f"tensor {name!r} not in specs"`). If nodes remain after the sweep, raise `ValueError(f"graph has a cycle involving {sorted(remaining_outputs)!r}")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_middleend.py -q`
Expected: PASS, 6/6.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 86 passed (80 existing + 6).

```bash
git add compiler/middleend.py tests/test_middleend.py
git commit -m "compiler: scheduler with deterministic topological order"
```

### Task 2: Graph passes

**Files:**
- Modify: `compiler/middleend.py`
- Test: `tests/test_middleend.py` (append)

**Interfaces:**
- Consumes: `Graph`, `Node`.
- Produces: `optimize(graph: Graph, weights: dict[str, np.ndarray]) -> tuple[Graph, dict[str, np.ndarray]]` — collapses `relu→relu` chains (keeps the first relu, drops the second, rewires the second's consumers to the first's output), prunes weights no node references, returns new objects (input dict untouched), idempotent.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from compiler.middleend import optimize


def test_collapse_double_relu():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="r1"),
                     Node(op="relu", inputs=("r1",), output="r2")))
    out, _ = optimize(g, {"w": np.ones((8, 8), dtype=np.int8)})
    assert [n.output for n in out.nodes] == ["c", "r1"]


def test_collapse_rewires_consumers_to_surviving_relu():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="r1"),
                     Node(op="relu", inputs=("r1",), output="r2"),
                     Node(op="requantize", inputs=("r2",), output="q")))
    out, _ = optimize(g, {"w": np.ones((8, 8), dtype=np.int8)})
    assert [n.output for n in out.nodes] == ["c", "r1", "q"]
    assert out.nodes[2].inputs == ("r1",)


def test_prune_weights():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    weights = {"w": np.ones((8, 8), dtype=np.int8),
               "orphan": np.zeros((4,), dtype=np.int32)}
    out, pruned = optimize(g, weights)
    assert set(pruned) == {"w"}
    assert set(weights) == {"w", "orphan"}  # input dict untouched


def test_optimize_fixed_point():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="r1"),
                     Node(op="relu", inputs=("r1",), output="r2"),
                     Node(op="relu", inputs=("r2",), output="r3")))
    w = {"w": np.ones((8, 8), dtype=np.int8)}
    once, w1 = optimize(g, w)
    twice, w2 = optimize(once, w1)
    assert twice == once and w2.keys() == w1.keys()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_middleend.py -q`
Expected: 6 pass, 4 errors — `ImportError: cannot import name 'optimize' from 'compiler.middleend'`.

- [ ] **Step 3: Implement `optimize` in `compiler/middleend.py`**

Iterate to a fixed point: find a `relu` node `B` whose single input is the output of a `relu` node `A` (only collapse when `len(B.inputs) == 1`); rebuild the node tuple dropping `B` and replacing every remaining node's reference to `B.output` with `A.output` (new `Node` objects — the dataclass is frozen). Then build the pruned weight dict: referenced names = union of all remaining nodes' inputs; return `{k: v for k, v in weights.items() if k in referenced}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_middleend.py -q`
Expected: PASS, 10/10.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 90 passed.

```bash
git add compiler/middleend.py tests/test_middleend.py
git commit -m "compiler: graph passes (relu collapse, dead-weight pruning)"
```

### Task 3: Memory planner

**Files:**
- Create: `compiler/memory.py`
- Test: `tests/test_memory.py` (new file)
- Modify: `docs/memory-map.md` (add planner section)

**Interfaces:**
- Consumes: `Node` (scheduled order from Task 1); `TensorSpec`.
- Produces:
  - `MemoryConfig(base: int = 0x1000, size: int = 0x100000, align: int = 64)` — frozen; `__post_init__` raises `ValueError` unless `base >= 0`, `size > 0`, `align > 0`, `base % align == 0`.
  - `Slot(address: int, size: int)` — frozen.
  - `MemoryPlan(slots: dict[str, Slot], peak_bytes: int, config: MemoryConfig)` — dataclass with `__eq__`.
  - `plan_memory(order: list[Node], specs: dict[str, TensorSpec], aliases: dict[str, str], config: MemoryConfig) -> MemoryPlan`.

Planner semantics (also written into `docs/memory-map.md`):

- Every tensor referenced by a node gets one slot; a tensor in `specs` that no node references gets none.
- Lifecycle is in scheduled node order. Per node: (1) births of inputs first used by this node, in `node.inputs` order (covers graph inputs and weights); (2) the output's birth if it is not aliased; (3) deaths of tensors whose last consuming node is this one. A slot is freed only when every tenant sharing it has died (aliases keep it alive).
- Aliased outputs share their input's slot (transitively resolved); the alias map for a chain is `{out: first_input}` for `relu` and `bias_add` nodes.
- Allocator: first-fit over freed blocks ordered by ascending address; an allocation span is `align_up(size, align)`; every slot address is `align`-aligned. Sizes in `Slot` are the logical byte sizes (dtype itemsize × element count); spans only affect packing.
- Peak and budget: `peak_bytes = max(address + size over slots) - config.base`; any slot overrunning `base + size` raises `ValueError(f"memory plan needs {peak} bytes, budget {config.size}")`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from compiler.graph import Node
from compiler.memory import MemoryConfig, plan_memory
from python.edge_npu.tensor import TensorSpec


def _chain():
    order = [Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("r", "b"), output="d"),
             Node(op="requantize", inputs=("d",), output="q")]
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(8,)),
             "d": TensorSpec(name="d", dtype="int32", shape=(8, 8)),
             "q": TensorSpec(name="q", dtype="int8", shape=(8, 8), scale=0.02)}
    aliases = {"r": "c", "d": "r"}
    return order, specs, aliases


def test_plan_worked_example_slots_and_peak():
    order, specs, aliases = _chain()
    plan = plan_memory(order, specs, aliases, MemoryConfig())
    got = {n: (slot.address, slot.size) for n, slot in plan.slots.items()}
    assert got == {"a": (0x1000, 64), "w": (0x1040, 64), "c": (0x1080, 256),
                   "r": (0x1080, 256), "b": (0x1000, 32), "d": (0x1080, 256),
                   "q": (0x1000, 64)}
    assert plan.peak_bytes == 0x180


def test_plan_alias_keeps_slot_alive_until_last_tenant():
    order, specs, aliases = _chain()
    plan = plan_memory(order, specs, aliases, MemoryConfig())
    assert plan.slots["r"] == plan.slots["c"] == plan.slots["d"]
    # q is born while d (a tenant of the c slot) is still live, so the c slot
    # is not reusable; b's freed block wins first-fit by address.
    assert plan.slots["q"].address == 0x1000


def test_plan_budget_exact_then_one_byte_short():
    order, specs, aliases = _chain()
    plan_memory(order, specs, aliases, MemoryConfig(size=0x180))  # exactly fits
    with pytest.raises(ValueError, match="384"):
        plan_memory(order, specs, aliases, MemoryConfig(size=0x17F))


def test_plan_aligns_every_address():
    order = [Node(op="matmul", inputs=("a", "w"), output="c")]
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(10, 10)),
             "w": TensorSpec(name="w", dtype="int8", shape=(10, 10)),
             "c": TensorSpec(name="c", dtype="int32", shape=(10, 10))}
    plan = plan_memory(order, specs, {}, MemoryConfig())
    assert plan.slots["a"].address == 0x1000
    assert plan.slots["w"].address == 0x1080
    assert plan.slots["c"].address == 0x1100
    assert all(s.address % 64 == 0 for s in plan.slots.values())


def test_plan_skips_unreferenced_tensors():
    order, specs, aliases = _chain()
    specs = dict(specs, ghost=TensorSpec(name="ghost", dtype="int8", shape=(8,)))
    plan = plan_memory(order, specs, aliases, MemoryConfig())
    assert "ghost" not in plan.slots


def test_plan_deterministic():
    order, specs, aliases = _chain()
    first = plan_memory(order, specs, aliases, MemoryConfig())
    second = plan_memory(order, specs, aliases, MemoryConfig())
    assert first == second


def test_config_validation():
    with pytest.raises(ValueError):
        MemoryConfig(align=0)
    with pytest.raises(ValueError):
        MemoryConfig(base=0x1001, align=64)
    with pytest.raises(ValueError):
        MemoryConfig(size=0)
```

Worked example derivation (the event rules applied to `_chain()`): node 0 births `a`@0x1000 (span 64), `w`@0x1040, `c`@0x1080 (span 256); `a` and `w` die after node 0. Node 1 aliases `r`→`c`. Node 2 births `b` (span 64) at first-fit 0x1000 and `d` aliases `c`. Node 3 births `q` (span 64) while `c`'s slot is still alive (tenants `c`,`r`,`d` — `d` dies at node 3), so `q` takes first-fit 0x1000; then `d` dies and frees the c slot. Peak = 0x1180 − 0x1000 = 0x180.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'compiler.memory'`.

- [ ] **Step 3: Implement `plan_memory` in `compiler/memory.py`**

Sweep nodes in order maintaining: `slot_of: dict[str, Slot]`, per-slot tenant sets with a death counter, `free: sorted list of (start, end)` aligned blocks, a bump pointer for fresh space, `last_use: dict[str, int]` (computed in one pass: last node index consuming each name; sinks get `len(order)`). Birth allocates first-fit from `free`, else fresh space at the bump pointer (aligned up). Death decrements the slot's live-tenant count and frees the span when it hits zero. Resolve alias targets transitively (follow `aliases` to a non-aliased root; validate the root has a slot). After the sweep, compute `peak_bytes` and enforce the budget.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory.py -q`
Expected: PASS, 7/7.

- [ ] **Step 5: Update `docs/memory-map.md`, full suite, commit**

Add a "Middle-end memory planner" section under "Executor model" with the semantics listed in this task's Interfaces block (slots, lifetimes, aliases, first-fit, alignment, budget error).

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 97 passed.

```bash
git add compiler/memory.py tests/test_memory.py docs/memory-map.md
git commit -m "compiler: liveness memory planner with named budget check"
```

### Task 4: Chain validation

**Files:**
- Modify: `compiler/middleend.py`
- Test: `tests/test_middleend.py` (append)

**Interfaces:**
- Consumes: scheduled order from Task 1.
- Produces: `validate(order: list[Node]) -> None` — three legality checks, each a `ValueError` naming what failed: fan-out (a node-produced tensor consumed by more than one node), output count (must be exactly one sink), and lowering support (every node's op has a lowering: `matmul`, `conv2d`, `relu`, `bias_add`, `max_pool`, `requantize`). Weights consumed by multiple nodes are not fan-out. An empty order fails the output-count check.

- [ ] **Step 1: Write the failing tests**

```python
from compiler.middleend import validate


def test_fanout_rejected():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("c", "b"), output="d"))
    specs = _specs("a", "w", "c", "r", "b", "d")
    order = schedule(Graph(nodes=nodes), specs)
    with pytest.raises(ValueError, match="'c'"):
        validate(order)


def test_shared_weight_is_not_fanout():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="bias_add", inputs=("c", "b"), output="d"),
             Node(op="bias_add", inputs=("d", "b"), output="e"))
    specs = _specs("a", "w", "c", "b", "d", "e")
    validate(schedule(Graph(nodes=nodes), specs))  # must not raise


def test_two_sinks_rejected():
    nodes = (Node(op="relu", inputs=("x",), output="r1"),
             Node(op="relu", inputs=("y",), output="r2"))
    specs = _specs("x", "r1", "y", "r2")
    with pytest.raises(ValueError, match="2"):
        validate(schedule(Graph(nodes=nodes), specs))


def test_unsupported_op_rejected():
    nodes = (Node(op="load", inputs=("x",), output="l"),)
    specs = _specs("x", "l")
    with pytest.raises(ValueError, match="load"):
        validate(schedule(Graph(nodes=nodes), specs))


def test_empty_graph_rejected():
    with pytest.raises(ValueError, match="0"):
        validate(schedule(Graph(nodes=()), {}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_middleend.py -q`
Expected: 10 pass, 5 errors — `ImportError: cannot import name 'validate'`.

- [ ] **Step 3: Implement `validate` in `compiler/middleend.py`**

One pass over `order`: collect `produced = {node.output}`; count consumers for names in `produced` (names not produced — weights and graph inputs — are exempt); raise on count > 1 naming the tensor. Collect sinks (`node.output` with zero consumers); raise unless exactly one, e.g. `f"graph must have a single output; found {len(sinks)}"`. Raise `f"no lowering for op {node.op!r}"` for ops outside the six supported ones (`load`/`store` pass `Node` validation but have no lowering).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_middleend.py -q`
Expected: PASS, 15/15.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 102 passed.

```bash
git add compiler/middleend.py tests/test_middleend.py
git commit -m "compiler: chain legality validation for lowering"
```

### Task 5: `lower_graph` driver

**Files:**
- Modify: `compiler/codegen.py` (address-explicit matmul emitter; no output change)
- Modify: `compiler/middleend.py`
- Test: `tests/test_middleend.py` (append)
- Modify: `docs/architecture.md` (add the middle-end stage)

**Interfaces:**
- Codegen: `check_matmul(node, specs) -> tuple[int, int, int, int, int, int]` (`m, n, k, a_bytes, w_bytes, c_bytes`, all existing validation); `emit_matmul(node, specs, a_addr, w_addr, c_addr) -> list[Command]` (LOAD, LOAD, MATMUL, STORE). `lower_matmul(node, specs, base)` is re-expressed as `check_matmul` + `emit_matmul` with contiguous addresses — identical output, proven by the existing `tests/test_codegen.py` (unmodified).
- Middleend: `LoweredGraph(graph: Graph, weights: dict[str, np.ndarray], cmds: list[Command], plan: MemoryPlan)`; `lower_graph(graph, specs, weights, attrs, *, memory: MemoryConfig | None = None) -> LoweredGraph`.

Driver flow: `optimize` → `schedule` → `validate` → build `aliases = {n.output: n.inputs[0] for n in order if n.op in ("relu", "bias_add")}` → `plan_memory` → emit per node → final STORE. Per-op dispatch: `matmul` → `emit_matmul` at the input/weight/output slots; `conv2d` → `[LOAD x@slot, LOAD w@slot] + lower_conv2d(base=slot(y))`; `relu`/`bias_add` → `lower_relu`/`lower_bias_add` at the output slot (the aliased input slot); `max_pool` → requires `attrs[node.output]` (`ValueError(f"pool node {out!r} has no attrs")`), `lower_max_pool(base=slot(out), size=..., stride=...)`; `requantize` → `lower_requantize(base=slot(out), scale=specs[out].scale)`. Final STORE: append `STORE` of the sink's slot and byte size unless the last command is already a STORE at that address and size (which keeps matmul-only streams byte-identical to `lower_matmul`).

- [ ] **Step 1: Write the failing tests**

```python
import struct

from compiler.binary import write_model
from compiler.codegen import emit_matmul, lower_matmul
from compiler.middleend import lower_graph
from python.edge_npu.commands import Command
from python.edge_npu.reference import bias_add, matmul_int8, relu, requantize
from python.edge_npu.sdk import Device


def _chain_graph():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("r", "b"), output="d"),
             Node(op="requantize", inputs=("d",), output="q"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(8,)),
             "d": TensorSpec(name="d", dtype="int32", shape=(8, 8)),
             "q": TensorSpec(name="q", dtype="int8", shape=(8, 8), scale=0.02)}
    return nodes, specs


def test_emit_matmul_matches_lower_matmul():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(64, 64)),
             "w": TensorSpec(name="w", dtype="int8", shape=(64, 64)),
             "c": TensorSpec(name="c", dtype="int32", shape=(64, 64))}
    assert emit_matmul(node, specs, 0x1000, 0x2000, 0x3000) == \
        lower_matmul(node, specs, base=0x1000)


def test_lower_graph_matmul_stream_equals_lower_matmul():
    import numpy as np
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(64, 64)),
             "w": TensorSpec(name="w", dtype="int8", shape=(64, 64)),
             "c": TensorSpec(name="c", dtype="int32", shape=(64, 64))}
    lg = lower_graph(Graph(nodes=(node,)), specs,
                     {"w": np.ones((64, 64), dtype=np.int8)}, {})
    assert lg.cmds == lower_matmul(node, specs, base=0x1000)
    assert lg.graph == Graph(nodes=(node,))


def test_lower_graph_chain_exact_stream():
    import numpy as np
    nodes, specs = _chain_graph()
    lg = lower_graph(Graph(nodes=nodes), specs,
                     {"w": np.ones((8, 8), dtype=np.int8),
                      "b": np.zeros((8,), dtype=np.int32)}, {})
    scale_bits = struct.unpack("<I", struct.pack("<f", 0.02))[0]
    assert lg.cmds == [
        Command(op="LOAD", address=0x1000, size=64),
        Command(op="LOAD", address=0x1040, size=64),
        Command(op="MATMUL", m=8, n=8, k=8),
        Command(op="STORE", address=0x1080, size=256),
        Command(op="RELU", address=0x1080, size=256),
        Command(op="BIAS_ADD", address=0x1080, size=256),
        Command(op="REQUANTIZE", address=0x1000, size=64, reserved=scale_bits, m=128),
        Command(op="STORE", address=0x1000, size=64),
    ]


def test_lower_graph_pool_requires_attrs():
    import numpy as np
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="max_pool", inputs=("c",), output="p"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 4, 4))}
    with pytest.raises(ValueError, match="'p'"):
        lower_graph(Graph(nodes=nodes), specs,
                    {"w": np.ones((8, 8), dtype=np.int8)}, {})


def test_lower_graph_pool_uses_attrs():
    import numpy as np
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="max_pool", inputs=("c",), output="p"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 4, 4))}
    lg = lower_graph(Graph(nodes=nodes), specs,
                     {"w": np.ones((8, 8), dtype=np.int8)},
                     {"p": {"size": 2, "stride": 2}})
    assert lg.cmds[-2:] == [
        Command(op="MAX_POOL", address=0x1180, size=512, m=2, n=2),
        Command(op="STORE", address=0x1180, size=512),
    ]


def test_lower_graph_end_to_end_matches_reference(tmp_path):
    import numpy as np
    nodes, specs = _chain_graph()
    rng = np.random.default_rng(11)
    w = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    b = rng.integers(-100, 100, size=(8,), dtype=np.int32)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w": w, "b": b}, {})
    path = str(tmp_path / "chain.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    expected = requantize(bias_add(relu(matmul_int8(x, w)), b), 0.02)
    np.testing.assert_array_equal(out, expected)


def test_lower_graph_empty_rejected():
    with pytest.raises(ValueError, match="0"):
        lower_graph(Graph(nodes=()), {}, {}, {})
```

(These import names match `python/edge_npu/reference.py` as already pinned by `tests/test_sdk.py`; hoist the new imports to the top of the file — they are shown here for locality.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_middleend.py -q`
Expected: 15 pass, 7 errors — `ImportError: cannot import name 'emit_matmul'` / `'lower_graph'`.

- [ ] **Step 3a: Refactor `compiler/codegen.py`**

Extract the existing matmul validation into `check_matmul(node, specs)` returning `(m, n, k, a_bytes, w_bytes, c_bytes)` (byte counts computed from spec dtypes, exactly as today). Add `emit_matmul(node, specs, a_addr, w_addr, c_addr)` building `[LOAD@a_addr, LOAD@w_addr, MATMUL, STORE@c_addr]` after calling `check_matmul`. Rewrite `lower_matmul(node, specs, base)` as `check_matmul` + `emit_matmul(node, specs, base, base + a_b, base + a_b + w_b)`. Do not touch the other lowerings.

- [ ] **Step 3b: Implement `LoweredGraph` and `lower_graph` in `compiler/middleend.py`**

As specified in Interfaces. `memory=None` means `MemoryConfig()`. Import the lowerings from `compiler.codegen` and `MemoryConfig`/`plan_memory` from `compiler.memory` (import at module top).

- [ ] **Step 3c: Update `docs/architecture.md`**

In the data-path diagram, insert the middle-end stage between "INT8 Graph + weights + specs" and "model.bin": `compiler/middleend.py` (schedule, optimize, validate, memory plan) lowers to the command stream; `compiler/binary.py` still packs. One or two lines; no other doc changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_middleend.py tests/test_codegen.py -q`
Expected: PASS — 22/22 in `test_middleend.py`, 18/18 in `test_codegen.py` (unmodified file proves the refactor changed nothing).

- [ ] **Step 5: Full suite + commit**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 109 passed.

```bash
git add compiler/codegen.py compiler/middleend.py tests/test_middleend.py docs/architecture.md
git commit -m "compiler: lower_graph driver with planned addresses"
```

### Task 6: Migrate demo and bench to `lower_graph`

**Files:**
- Modify: `demo/demo_inference.py` (all four panels)
- Modify: `benchmarks/bench_matmul.py`

**Interfaces:**
- Consumes: `lower_graph` and `LoweredGraph` from Task 5.
- Produces: no new API. The acceptance evidence is (a) the four demo panel stdout bytes are identical to before the change, and (b) `bash scripts/check.sh` prints `check ok`. The panels' numeric assertions and printed dashboard stay exactly as they are.

- [ ] **Step 1: Capture the before-picture**

Run: `mkdir -p .superpowers/sdd/2026-10-10-middleend-core && bash -c 'unset PYTHONPATH; export PATH=/c/MinGW/bin:$PATH; .venv/Scripts/python.exe demo/demo_inference.py' > .superpowers/sdd/2026-10-10-middleend-core/demo_before.txt`
Expected: exit 0; file contains the four dashboards.

- [ ] **Step 2: Rewrite the call sites**

- `main()`: `lg = lower_graph(Graph(nodes=(node,)), specs, {"w": bq}, {})`; `write_model(path, lg.graph, lg.weights, lg.cmds)`. Remove the now-unused `lower_matmul` import if nothing else uses it.
- `_demo_chain()`: give `q`'s spec `scale=scale`; replace the hand-assembled `cmds = lower_matmul(...) + ... + [Command(op="STORE", ...)]` with `lg = lower_graph(Graph(nodes=nodes), specs, {"w": w, "b": b}, {})`; write via `lg`.
- `_demo_tinycnn()`: `lg = lower_graph(Graph(nodes=nodes), specs, {"w": w}, {"p": {"size": 2, "stride": 2}})`; write via `lg`; drop the manual LOADs and STORE.
- `_demo_onnx_pipeline()`: capture the `attrs` return of `load_onnx` (currently `_`); `lg = lower_graph(graph, qspecs, qw, attrs)`; write via `lg`.
- `benchmarks/bench_matmul.py` `_edgenpu_sim()`: `lg = lower_graph(Graph(nodes=(node,)), specs, {"w": bq}, {})`; `write_model(path, lg.graph, lg.weights, lg.cmds)`.

- [ ] **Step 3: Prove panel output unchanged**

Run: `bash -c 'unset PYTHONPATH; export PATH=/c/MinGW/bin:$PATH; .venv/Scripts/python.exe demo/demo_inference.py' > .superpowers/sdd/2026-10-10-middleend-core/demo_after.txt; diff .superpowers/sdd/2026-10-10-middleend-core/demo_before.txt .superpowers/sdd/2026-10-10-middleend-core/demo_after.txt && echo PANELS-IDENTICAL`
Expected: `PANELS-IDENTICAL`, no diff lines.

- [ ] **Step 4: Run the full gate**

Run: `bash scripts/check.sh`
Expected: pytest green (109), self-check FAIL line as designed, `runtime tests ok`, `check ok`.

- [ ] **Step 5: Commit**

```bash
git add demo/demo_inference.py benchmarks/bench_matmul.py
git commit -m "demo/bench: assemble streams through lower_graph"
```
