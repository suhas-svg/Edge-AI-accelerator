# Tiling (pad-and-trim) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Legalize odd-dim graphs by padding to tile multiples at compile time and trimming at the model boundary, so every emitted command stays tile-aligned and every odd shape runs bit-exact.

**Architecture:** New `legalize(order, specs, attrs)` pass in `compiler/middleend.py` runs inside `lower_graph` after scheduling: it verifies logical shape consistency (same messages as lowering), then rewrites specs to padded shapes via forward propagation. Weights stay logical in model.bin; the SDK pads inputs and weights with zeros to the command LOAD sizes at predict entry, runs the existing pipeline untouched on padded buffers, and trims the final buffer to the propagated logical sink shape. Zero-padding is exactly correct: matmul/conv zero entries contribute nothing, elementwise is pointwise, and every logical pool output window lies fully inside the logical input (max read index `((H-ws)//s)*s+ws-1 <= H-1`).

**Tech Stack:** Python 3.12 (`.venv`), numpy, pytest; gcc checks via `scripts/check.sh`.

**Spec:** the user-confirmed pad-and-trim design (second PR of the two-PR split; commands stay tile-aligned; SDK computes padded and trims at the boundary) plus the in-repo contracts `docs/architecture.md`, `docs/memory-map.md`, `docs/command-format.md`, `docs/model-format.md`, `docs/runtime-api.md`. No standalone tiling spec section is reachable in-repo; rulings made while executing are provisional against those contracts.

## Global Constraints

- Environment repair is mandatory before any python/gcc run: `/c/MinGW/bin` first on `PATH` and `PYTHONPATH` unset. `bash scripts/check.sh` does both; for direct runs use `.venv/Scripts/python.exe -m pytest`.
- Contracts frozen: no edits to `docs/command-format.md` or `docs/model-format.md`. Command encoding is unchanged (padded dims ride the existing m/n/k/size fields); model.bin still stores logical weights.
- `Node` and `Graph` stay as they are; pool params stay in the frontend `attrs` side-table.
- Existing test files must not be modified — they pin the behavior this work must preserve (in particular the codegen odd-dim rejections stay as backstop). New tests go in `tests/test_tiling.py`.
- Every task ends green on `.venv/Scripts/python.exe -m pytest tests/ -q`; the branch ends green on `bash scripts/check.sh` ("check ok").

## Review Focus

- Already-aligned graphs: legalize is identity on tile-aligned specs, so pre-tiling `lower_graph` streams are byte-identical. Pinned in Task 1.
- Shared padded dims: matmul inner dims and conv channels pad jointly from the max; a logical mismatch still raises the inner-dim/channel message on logical values, never silently pads into agreement. Pinned in Task 1.
- Tiny shapes: a 1x8 row tiles to 8x8 and trims back to exactly (1, 8). Pinned in Task 3.
- Pool exactness on padded inputs: logical pool windows never read padding, so padded-then-trimmed pool equals logical pool. Pinned in Task 3 by an odd-spatial conv→relu→pool e2e vs reference.
- Peak and budget are measured on the padded plan (peak grows with padding); the exact-fit boundary still passes and one-byte-short still names both numbers. Pinned in Task 2.

**Out of scope:** ragged edge tiles (rejected in favor of pad-and-trim), C runtime changes (padded dims already satisfy its tile checks), demo/bench changes (4 panels untouched), fusion/double-buffering (later items).

---

### Task 1: `legalize` pass

**Files:**
- Modify: `compiler/middleend.py`
- Test: `tests/test_tiling.py` (new file)

**Interfaces:**
- Consumes: scheduled `order: list[Node]`; `specs`; `attrs: dict[str, dict]`.
- Produces: `legalize(order: list[Node], specs: dict[str, TensorSpec], attrs: dict[str, dict]) -> dict[str, TensorSpec]` — new dict, input untouched. Rules below. Dtypes, scales, zero_points pass through unchanged.

Legalize rules (forward pass in scheduled order; per tensor track logical shape from specs and padded shape being built):

- Sources (tensors first appearing as a node input that no earlier node produced — graph inputs and weights), rank >= 2: pad every dim up to a multiple of 8. Rank-1 sources stay logical here (bias is resized by its node's rule; a rank-1 tensor feeding anything else directly is padded by the consumer's rule or, for a rank-1 elementwise target, by extending its last dim in steps of 8 until numel % 64 == 0).
- `matmul`: verify ranks are 2, `ka == kb` else `ValueError(f"inner dim mismatch {ka} vs {kb}")`, output `== (m, n)` else the existing output-shape message — all on logical shapes. Padded: `m'`, `k' = ceil(max(ka,kb)/8)*8`, `n'` rounded up; rewrite both inputs (shared `k'`) and the output.
- `conv2d`: verify ranks (3/4/3), `kc == c` else channel message, square kernel, output `== (k, oh, ow)` else output message — logical. Padded: `K'`, `C' = ceil(max(c,kc)/8)*8` rounded up; spatial pads the OUTPUT: `OH'`, `OW'` rounded up, `H' = OH' + KH - 1`, `W' = OW' + KW - 1` (kernel dims never padded). Rewrite input, weights (C dim only), output.
- `relu` / `requantize`: verify input shape == output shape (same messages as lowering). Padded output inherits the input's padded shape exactly (required for in-place aliasing).
- `bias_add`: verify data == output and bias broadcast (same messages as lowering, logical). Padded output inherits padded data shape. Bias: if 1D matching the logical last dim, rewrite to `(padded_last,)`; if full-shape, inherit padded data shape.
- `max_pool`: requires `attrs[node.output]` else `ValueError(f"pool node {out!r} has no attrs")`. Verify output equals pool(logical input) with attrs window/stride (same message as lowering). Padded output = pool(padded input); logical output = pool(logical input).
- Unreferenced spec entries pass through unchanged.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from compiler.graph import Graph, Node
from compiler.middleend import legalize, schedule
from python.edge_npu.tensor import TensorSpec


def _mm_specs(a, w, c):
    return {"a": TensorSpec(name="a", dtype="int8", shape=a),
            "w": TensorSpec(name="w", dtype="int8", shape=w),
            "c": TensorSpec(name="c", dtype="int32", shape=c)}


def _order(nodes, specs):
    return schedule(Graph(nodes=nodes), specs)


def test_legalize_aligned_is_identity():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),)
    specs = _mm_specs((64, 64), (64, 64), (64, 64))
    assert legalize(_order(nodes, specs), specs, {}) == specs


def test_legalize_odd_matmul_pads_to_tiles():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),)
    specs = _mm_specs((60, 64), (64, 48), (60, 48))
    got = legalize(_order(nodes, specs), specs, {})
    assert got["a"].shape == (64, 64)
    assert got["w"].shape == (64, 48)
    assert got["c"].shape == (64, 48)
    assert (got["a"].dtype, got["c"].dtype) == ("int8", "int32")


def test_legalize_inner_mismatch_still_rejected():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),)
    specs = _mm_specs((8, 7), (9, 8), (8, 8))
    with pytest.raises(ValueError, match="inner dim mismatch 7 vs 9"):
        legalize(_order(nodes, specs), specs, {})


def test_legalize_conv_pads_output_spatial():
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="y"),)
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 9, 9)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "y": TensorSpec(name="y", dtype="int32", shape=(8, 7, 7))}
    got = legalize(_order(nodes, specs), specs, {})
    assert got["x"].shape == (8, 10, 10)
    assert got["w"].shape == (8, 8, 3, 3)
    assert got["y"].shape == (8, 8, 8)


def test_legalize_elementwise_inherits_padded_shape():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r")))
    specs = dict(_mm_specs((60, 64), (64, 64), (60, 64)),
                 r=TensorSpec(name="r", dtype="int32", shape=(60, 64)))
    got = legalize(_order(nodes, specs), specs, {})
    assert got["c"].shape == got["r"].shape == (64, 64)


def test_legalize_pads_bias_to_padded_last_dim():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="bias_add", inputs=("c", "b"), output="d")))
    specs = dict(_mm_specs((60, 60), (60, 60), (60, 60)),
                 b=TensorSpec(name="b", dtype="int32", shape=(60,)),
                 d=TensorSpec(name="d", dtype="int32", shape=(60, 60)))
    got = legalize(_order(nodes, specs), specs, {})
    assert got["d"].shape == (64, 64)
    assert got["b"].shape == (64,)


def test_legalize_pool_derives_both_shapes():
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="c"),
             Node(op="max_pool", inputs=("c",), output="p")))
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 9, 9)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 7, 7)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 3, 3))}
    got = legalize(_order(nodes, specs), specs, {"p": {"size": 2, "stride": 2}})
    assert got["c"].shape == (8, 8, 8)
    assert got["p"].shape == (8, 4, 4)


def test_legalize_pool_needs_attrs():
    nodes = (Node(op="max_pool", inputs=("x",), output="p"),)
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 8, 8)),
             "p": TensorSpec(name="p", dtype="int8", shape=(8, 4, 4))}
    with pytest.raises(ValueError, match="'p'"):
        legalize(_order(nodes, specs), specs, {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tiling.py -q`
Expected: collection error — `ImportError: cannot import name 'legalize' from 'compiler.middleend'`.

- [ ] **Step 3: Implement `legalize` in `compiler/middleend.py`**

One forward pass over `order` as specified in Interfaces. Build fresh `TensorSpec` objects (same name/dtype/scale/zero_point, new shape) into a new dict; never mutate the input. Produced-set tracking distinguishes sources from intermediates (a tensor is a source if no earlier node produced it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tiling.py -q`
Expected: PASS, 8/8.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 117 passed (109 existing + 8).

```bash
git add compiler/middleend.py tests/test_tiling.py
git commit -m "compiler: legalize pass pads odd dims to tile multiples"
```

### Task 2: Wire legalize into `lower_graph`

**Files:**
- Modify: `compiler/middleend.py`
- Test: `tests/test_tiling.py` (append)

**Interfaces:**
- Consumes: `legalize` from Task 1.
- Produces: `lower_graph` flow becomes optimize → schedule → legalize → validate → plan → emit. The plan, slots, and stream all reflect padded shapes; weights stay logical. No signature change.

- [ ] **Step 1: Write the failing tests**

```python
from compiler.middleend import lower_graph
from python.edge_npu.commands import Command


def test_lower_graph_odd_matmul_stream_is_tiled():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = _mm_specs((24, 16), (16, 32), (24, 32))
    lg = lower_graph(Graph(nodes=(node,)), specs,
                     {"w": np.ones((16, 32), dtype=np.int8)}, {})
    assert lg.cmds == [
        Command(op="LOAD", address=0x1000, size=512),
        Command(op="LOAD", address=0x1200, size=512),
        Command(op="MATMUL", m=32, n=32, k=16),
        Command(op="STORE", address=0x1400, size=4096),
    ]
    assert lg.weights["w"].shape == (16, 32)  # stored logical


def test_lower_graph_padded_budget_boundary():
    from compiler.memory import MemoryConfig
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = _mm_specs((24, 16), (16, 32), (24, 32))
    weights = {"w": np.ones((16, 32), dtype=np.int8)}
    lower_graph(Graph(nodes=(node,)), specs, weights, {},
                memory=MemoryConfig(size=5120))  # padded peak exactly
    with pytest.raises(ValueError, match="5120"):
        lower_graph(Graph(nodes=(node,)), specs, weights, {},
                    memory=MemoryConfig(size=5119))


def test_lower_graph_odd_dims_still_rejected_below_legalize():
    from compiler.codegen import lower_matmul
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    with pytest.raises(ValueError, match="not divisible"):
        lower_matmul(node, _mm_specs((60, 64), (64, 64), (60, 64)), base=0x1000)
```

LOAD sizes are padded bytes: `a` pads (24,16)→(32,16) = 512 bytes @0x1000 (span 512); `w` (16,32) = 512 @0x1200; `c` (32,32) INT32 = 4096 @0x1400. Padded peak = 0x1400 + 4096 − 0x1000 = 5120.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tiling.py -q`
Expected: 8 pass, 2 fail — the stream and budget tests fail on the div-8 rejection (`ValueError: dim m=24 not divisible by 8`); the backstop test passes.

- [ ] **Step 3: Insert legalize into `lower_graph`**

After `order = schedule(...)`, add `specs = legalize(order, specs, attrs)` before `validate(order)`. Nothing else changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tiling.py -q`
Expected: PASS, 11/11.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 120 passed.

```bash
git add compiler/middleend.py tests/test_tiling.py
git commit -m "compiler: lower through legalized padded specs"
```

### Task 3: SDK pad-at-entry, trim-at-exit

**Files:**
- Modify: `python/edge_npu/sdk.py`
- Test: `tests/test_tiling.py` (append)

**Interfaces:**
- Consumes: `LoweredGraph` streams with padded dims and logical weights; runtime inputs at logical shapes.
- Produces: unchanged `predict` signature. Entry pads `x` and `w` with zeros to the command LOAD sizes and reshapes to the padded geometry (matmul: `(M.cmd, K.cmd)` / `(K.cmd, N.cmd)`; conv: `(C', H', W')` / `(K', C', KH, KW)` derived from LOAD sizes, the CONV2D dims, and logical kernel dims — with `C' = load_bytes / (K'*KH*KW)`). The existing pipeline then runs untouched on padded buffers. Exit trims the final buffer to the logical sink shape propagated from logical input/weight shapes (pool windows/strides from the MAX_POOL commands) and returns the trimmed buffer. All existing command checks keep passing because they compare padded against padded.

- [ ] **Step 1: Write the failing tests**

```python
from compiler.binary import write_model
from python.edge_npu.reference import bias_add, matmul_int8, relu
from python.edge_npu.sdk import Device


def test_odd_matmul_end_to_end_matches_reference(tmp_path):
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = _mm_specs((24, 16), (16, 32), (24, 32))
    rng = np.random.default_rng(21)
    w = rng.integers(-128, 127, size=(16, 32), dtype=np.int8)
    lg = lower_graph(Graph(nodes=(node,)), specs, {"w": w}, {})
    path = str(tmp_path / "odd.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(24, 16), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    assert out.shape == (24, 32)
    np.testing.assert_array_equal(out, matmul_int8(x, w))


def test_tiny_row_tiles_and_trims(tmp_path):
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = _mm_specs((1, 8), (8, 8), (1, 8))
    rng = np.random.default_rng(22)
    w = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    lg = lower_graph(Graph(nodes=(node,)), specs, {"w": w}, {})
    path = str(tmp_path / "tiny.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(1, 8), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    assert out.shape == (1, 8)
    np.testing.assert_array_equal(out, matmul_int8(x, w))


def test_odd_conv_pool_matches_reference(tmp_path):
    from python.edge_npu.reference import conv2d_int8, max_pool
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="max_pool", inputs=("r",), output="p"))
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(4, 9, 9)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 4, 3, 3)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 7, 7)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 7, 7)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 3, 3))}
    rng = np.random.default_rng(23)
    w = rng.integers(-128, 127, size=(8, 4, 3, 3), dtype=np.int8)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w": w},
                     {"p": {"size": 2, "stride": 2}})
    path = str(tmp_path / "oddcnn.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(4, 9, 9), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    assert out.shape == (8, 3, 3)
    np.testing.assert_array_equal(out, max_pool(relu(conv2d_int8(x, w)), 2, 2))
```

Padded geometry of the conv test: C `max(4,4)=4→8`, K 8, OH `7→8` so H' 10, OW likewise; padded `x(8,10,10) w(8,8,3,3) c(8,8,8)`, padded pool `(8,4,4)`, trimmed to `(8,3,3)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tiling.py -q`
Expected: 11 pass, 3 fail — LOAD-size mismatches (padded command sizes vs logical `nbytes`).

- [ ] **Step 3: Implement pad/trim in `python/edge_npu/sdk.py`**

Pad flat inputs/weights with zeros to the LOAD sizes and reshape to the padded geometry from the command dims (matmul uses the MATMUL m/n/k; conv derives `C'`, `H' = OH'+KH-1`, `W' = OW'+KW-1` with kernel dims from the logical weights). Keep every existing check and the whole padded pipeline as-is. After the final STORE check, trim to the logical sink shape (propagated from the logical input/weight shapes through the chain; pool steps read window/stride from the MAX_POOL commands) and return the trimmed buffer.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tiling.py tests/test_sdk.py -q`
Expected: PASS — 14/14 in `test_tiling.py`, 12/12 in `test_sdk.py` (unmodified file proves aligned-model behavior is unchanged).

- [ ] **Step 5: Full suite + commit**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 123 passed.

```bash
git add python/edge_npu/sdk.py tests/test_tiling.py
git commit -m "sdk: pad to tile geometry at entry, trim to logical shape at exit"
```

### Task 4: Docs, gate, commit

**Files:**
- Modify: `docs/memory-map.md`, `docs/architecture.md`, `docs/runtime-api.md`

**Interfaces:**
- Consumes: the behavior built in Tasks 1–3.
- Produces: no code change. `memory-map.md` Alignment discipline: tile multiples are now guaranteed by middle-end legalization (codegen rejections stay as backstop); peak/budget are measured on padded bytes. `architecture.md`: legalize joins the middle-end stage line. `runtime-api.md`: cycles count executed (padded) MACs; trimming is exact so numerics match the logical reference.

- [ ] **Step 1: Update the three docs**

One section each; no version bumps, no contract edits.

- [ ] **Step 2: Run the full gate**

Run: `bash scripts/check.sh`
Expected: pytest green (123), self-check FAIL line as designed, `runtime tests ok`, `check ok`.

- [ ] **Step 3: Commit**

```bash
git add docs/memory-map.md docs/architecture.md docs/runtime-api.md
git commit -m "docs: tiling alignment, padded cycles, trimmed outputs"
```
