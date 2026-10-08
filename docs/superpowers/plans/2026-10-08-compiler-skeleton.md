# Compiler skeleton implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the compiler skeleton that lowers a matmul graph to an EdgeNPU command stream and packs it into a versioned model.bin.

**Architecture:** Graph of frozen dataclass nodes lowers through codegen to LOAD, MATMUL, STORE commands, then a binary writer packs header, tensors, weights, and command bytes. Each layer parses once and returns typed objects.

**Tech Stack:** Python 3.10+, numpy, struct, pytest.

**Spec:** docs/command-format.md, docs/tensor-format.md, docs/model-format.md, python/edge_npu/commands.py, python/edge_npu/tensor.py

## Global Constraints

- INT8 inputs and weights, INT32 accumulators.
- Command record is 19 bytes LE, header is 6 bytes LE, opcodes are LOAD=0x01, MATMUL=0x02, STORE=0x03.
- Tensor dtypes are fp32, int8, int32 with positive scale.
- Every task ends green: `.venv/Scripts/python.exe -m pytest tests/ -q` plus the gcc compile in `bash scripts/check.sh`.
- Shell is Windows Git Bash. Repo python is `.venv/Scripts/python.exe`.

## Review Focus

- Matmul dims not divisible by 8 reach codegen, and a reasonable person expects a named rejection, not silent tiling. Pinned by the rejection test in Task 2.
- A truncated model.bin reaches the reader, and a reasonable person expects an error naming the short section. Pinned by the truncation test in Task 3.
- An unknown model version reaches the reader, and a reasonable person expects an error carrying the version number. Pinned by the version test in Task 3.
- Duplicate tensor names reach the graph, and a reasonable person expects construction to fail. Pinned by the duplicate test in Task 1.
- Out-of-range INT8 values reach the packer, and a reasonable person expects a rejection before bytes are written. Pinned by the range test in Task 3.

---

### Task 1: Graph types

**Files:**
- Create: `compiler/__init__.py`
- Create: `compiler/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `TensorSpec` from `python/edge_npu/tensor.py` for shape and dtype facts.
- Produces: `Node(op: OpKind, inputs: tuple[str, ...], output: str)` and `Graph(nodes: tuple[Node, ...])` with `Graph.order() -> list[Node]`. Constructor raises `ValueError` on duplicate outputs or unknown op strings.

- [ ] **Step 1: Write the failing tests**

```python
def test_two_node_chain_orders():
    g = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="y")))
    assert [n.output for n in g.order()] == ["c", "y"]

def test_duplicate_output_raises():
    with pytest.raises(ValueError):
        Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                     Node(op="relu", inputs=("c",), output="c")))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph.py -v`
Expected: FAIL with "Graph not defined" or "Node not defined".

- [ ] **Step 3: Implement `Node` and `Graph` in `compiler/graph.py`**

Frozen dataclasses. `OpKind` is the `Literal` of op strings from `python/edge_npu/tensor.py`, checked at runtime against the tuple `("matmul", "relu", "bias_add", "conv2d", "max_pool", "requantize", "load", "store")`. `order()` returns nodes in tuple order for now.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_graph.py -v`
Expected: PASS, 2/2.

- [ ] **Step 5: Commit**

```bash
git add compiler/__init__.py compiler/graph.py tests/test_graph.py
git commit -m "compiler: add graph types with ordering"
```

### Task 2: Matmul codegen

**Files:**
- Create: `compiler/codegen.py`
- Test: `tests/test_codegen.py`

**Interfaces:**
- Consumes: `Node` and `Graph` from Task 1, `TensorSpec` shapes, `Command` from `python/edge_npu/commands.py`.
- Produces: `lower_matmul(node: Node, specs: dict[str, TensorSpec], base: int) -> list[Command]`. Layout rule: input LOAD at `base`, weight LOAD at `base + input_bytes`, MATMUL with m, n, k from shapes, STORE of INT32 output at `base + input_bytes + weight_bytes`. Sizes are byte counts. Rejects dims not divisible by 8 with `ValueError` naming the dim.

- [ ] **Step 1: Write the failing tests**

```python
def test_lower_64x64_matmul():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(64, 64)),
             "w": TensorSpec(name="w", dtype="int8", shape=(64, 64)),
             "c": TensorSpec(name="c", dtype="int32", shape=(64, 64))}
    assert lower_matmul(node, specs, base=0x1000) == [
        Command(op="LOAD", address=0x1000, size=4096),
        Command(op="LOAD", address=0x2000, size=4096),
        Command(op="MATMUL", m=64, n=64, k=64),
        Command(op="STORE", address=0x3000, size=16384),
    ]

def test_odd_dim_rejected():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(60, 64)),
             "w": TensorSpec(name="w", dtype="int8", shape=(64, 64)),
             "c": TensorSpec(name="c", dtype="int32", shape=(60, 64))}
    with pytest.raises(ValueError):
        lower_matmul(node, specs, base=0x1000)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_codegen.py -v`
Expected: FAIL with "lower_matmul not defined".

- [ ] **Step 3: Implement `lower_matmul` in `compiler/codegen.py`**

Read m, k from the input shape and k, n from the weight shape. Raise `ValueError` unless the inner dims agree and m, n, k are each divisible by 8.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_codegen.py -v`
Expected: PASS, 2/2.

- [ ] **Step 5: Commit**

```bash
git add compiler/codegen.py tests/test_codegen.py
git commit -m "compiler: lower matmul to LOAD MATMUL STORE"
```

### Task 3: model.bin writer and reader

**Files:**
- Create: `compiler/binary.py`
- Test: `tests/test_binary.py`

**Interfaces:**
- Consumes: `Graph` from Task 1, `list[Command]` from Task 2, `weights: dict[str, np.ndarray]`.
- Produces: `MODEL_VERSION: int = 1`, `ModelPack(version: int, graph: Graph, weights: dict[str, np.ndarray], cmds: list[Command])`, `write_model(path: str, graph: Graph, weights: dict[str, np.ndarray], cmds: list[Command]) -> None`, `read_model(path: str) -> ModelPack`. Format: header magic u16 `0x454D` plus version u16, tensor count u32, command-bytes length u32, all LE. Per tensor: name length u8, name bytes, dtype u8 with 0=fp32 1=int8 2=int32, rank u8, dims u32 each, scale f32, zero point i8. Then weight bytes in tensor order, then command bytes. Reader raises `ValueError` on bad magic, unknown version, or any short section.

- [ ] **Step 1: Write the failing tests**

```python
def test_roundtrip_single_matmul(tmp_path):
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    weights = {"w": np.ones((8, 8), dtype=np.int8)}
    cmds = [Command(op="LOAD", address=0x1000, size=64),
            Command(op="MATMUL", m=8, n=8, k=8),
            Command(op="STORE", address=0x3000, size=256)]
    path = str(tmp_path / "tiny.bin")
    write_model(path, graph, weights, cmds)
    pack = read_model(path)
    assert pack.version == 1
    assert pack.graph == graph
    assert pack.weights["w"].tolist() == weights["w"].tolist()
    assert pack.cmds == cmds

def test_unknown_version_raises(tmp_path):
    path = str(tmp_path / "bad.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<HHII", 0x454D, 99, 0, 0))
    with pytest.raises(ValueError):
        read_model(path)

def test_truncated_raises(tmp_path):
    path = str(tmp_path / "short.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<HHII", 0x454D, 1, 1, 0))
    with pytest.raises(ValueError):
        read_model(path)

def test_out_of_range_weight_raises(tmp_path):
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    weights = {"w": np.array([[200]], dtype=np.int16)}
    with pytest.raises(ValueError):
        write_model(str(tmp_path / "o.bin"), graph, weights, [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_binary.py -v`
Expected: FAIL with "write_model not defined".

- [ ] **Step 3: Implement `write_model` and `read_model` in `compiler/binary.py`**

Validate every dtype value fits its declared range before writing any byte. Validate each section length on read before unpacking it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_binary.py -v`
Expected: PASS, 4/4.

- [ ] **Step 5: Commit**

```bash
git add compiler/binary.py tests/test_binary.py
git commit -m "compiler: add versioned model.bin writer and reader"
```
