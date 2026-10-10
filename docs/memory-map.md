# EdgeNPU Memory Map

Software-side layout rules. These are the addresses codegen emits and the
executor checks; the hardware memory map (BRAM windows, DMA addresses)
remains the partner's to document against this.

## Linear layout rule (`lower_matmul`)

Given `base`:

| Region | Address | Bytes |
|---|---|---|
| Input | `base` | m·k·dtype |
| Weights | `base + input_bytes` | k·n·dtype |
| Output | `base + input_bytes + weight_bytes` | m·n·4 (INT32) |

Elementwise and conv ops take an explicit `base` for their target buffer
and emit `address + size` for the output in bytes:

| Op | Output bytes |
|---|---|
| RELU, BIAS_ADD | elements·4 (INT32) |
| CONV2D | K·OH·OW·4 (INT32) |
| MAX_POOL | C·OH·OW·dtype |
| REQUANTIZE | elements·1 (INT8) |

## Executor model

### Middle-end memory planner (`compiler/memory.py`)

`plan_memory` assigns one slot (address + logical byte size) per tensor a
node references; tensors in `specs` that no node references get no slot.
Lifetimes run in scheduled node order. Per node: (1) births of inputs first
used here, in `node.inputs` order; (2) the output's birth unless aliased;
(3) deaths of tensors whose last consuming node is this one. Aliased
outputs (`relu`/`bias_add` map `{out: first_input}`) share their input's
slot, and the slot frees only after every tenant sharing it has died.
Allocation is first-fit over freed blocks in ascending address order;
spans round up to the alignment (default 64) while `Slot.size` keeps the
logical bytes. Exceeding the budget raises
`ValueError("memory plan needs {peak} bytes, budget {size}")`.

The SDK executor keys buffers by stream order, not by address decoding:
LOAD sizes must equal input then weight bytes in order; each compute
command's shape fields and byte size must match the live buffer; the final
STORE must match the output bytes. Intermediate STOREs are permitted; only
the last one is checked against the result.

## Alignment discipline

Tile multiples are guaranteed by middle-end legalization (`legalize` pads
odd dims to multiples of 8 and the SDK trims to the logical shape at the
model boundary), so every emitted command is tile-aligned:

- MATMUL dims m/n/k: each divisible by 8.
- CONV2D dims K/C/OH/OW: each divisible by 8.
- Elementwise/pool/requantize: total elements divisible by 64.
- The per-op lowering checks for the above stay in place as backstop and
  still reject unlegalized odd dims with named errors.
- Peak and budget are measured on padded bytes.
- The C runner (`enpu_run_matmul`) additionally requires buffers sized to
  at least m·k, k·n, and m·n·4 bytes respectively.

## Test that pins it

`tests/test_codegen.py` (layout values and every rejection),
`tests/test_sdk.py` (stream-vs-buffer checks), `runtime/tests/test_runtime.c`
(buffer sizing).
