# EdgeNPU Hardware Bring-Up

Day-one procedure for the hardware partner bringing RTL onto the
Artix-7 board. The software lane is complete and frozen; this doc is
the path from "board on desk" to the first hardware/software vertical
slice.

## Prerequisites

- Artix-7 board (35T minimum; 100T comfortable), USB/UART cable.
- Xilinx Vivado WebPACK (free; supports the whole Artix-7 family).
- Host env: this repo's `.venv` (Python 3.12), `/c/MinGW/bin` first on
  `PATH`, `PYTHONPATH` unset — `scripts/check.sh` already does both.

## Frozen contracts (build to these, not around them)

| Contract | Version | Doc |
|---|---|---|
| Command format | v0.4 (8 opcodes) | `docs/command-format.md` |
| Model binary | v0.1 | `docs/model-format.md` |
| Register map | draft, needs your signoff | `docs/register-map.md` |
| Memory map | as documented | `docs/memory-map.md` |
| Tensor/vector formats | as documented | `docs/tensor-format.md`, `docs/vector-format.md` |

Breaking-change rule: any RTL-driven contract question becomes a
versioned proposal (v0.5), never a silent edit.

## Step 1 — Golden vectors

```bash
.venv/Scripts/python.exe tools/gen_vectors.py
```

Writes deterministic INT8 vectors to `tests/vectors/` (tile-aligned
only). Odd-shape vectors live in `tests/vectors_odd/` and are
software-side only — the RTL harness glob never sees them, so never
pad or trim on the hardware side; the compiler pads, the SDK trims.

## Step 2 — Verify RTL against the goldens

1. Copy `tests/vectors/matmul_*.npz` to a work directory (do not
   modify the originals).
2. Run the testbench; for each vector, write the hardware result back
   into its copy under the key `rtl_out` (same shape and dtype as
   `expected`).
3. Run the harness against the work directory:

```bash
.venv/Scripts/python.exe tools/verify_rtl.py <work-dir>
```

Exit 0 means every vector matches bit-exact. A vector without the
`rtl_out` key counts as a failure, so a half-finished testbench run
cannot silently pass. The harness itself is proven honest:
`scripts/check.sh` and `tests/test_verify_rtl.py` run it against a
deliberately corrupted copy (built by `tools/selfcheck.py`) and demand
failure — so a green run here means the harness can actually catch a
mismatch, not that it passes everything.

Gate: no integration proceeds until this is green.

## Step 3 — Synthesis and timing

Synthesize for the Artix-7 target; typical expectation is ~100 MHz
fabric clock for the 8x8 INT8 array (~6.4 GOPS INT8 peak). Record the
achieved frequency and resource usage (LUT/DSP/BRAM %) — the software
benchmark needs the real clock, and the dashboard needs the real
utilization.

## Step 4 — Register-map signoff

`docs/register-map.md` is draft. Confirm widths, reset values, and
IRQ behavior, then sign off in the doc. This unblocks the driver.

## Step 5 — Transport decision

Write down how the host moves the model binary and command stream
onto the board (UART, JTAG, AXI — your call). A kernel driver comes
later; a userspace path is enough for the first slice.

## Step 6 — First vertical slice

Target (spec §44): INT8 64x64 matrix multiplication, host program to
board and back:

```text
Host → Runtime → Transport → DMA → FPGA memory →
8x8 MAC array → FPGA memory → DMA → Host
```

The result must match `python/edge_npu/reference.py` bit-exact, and
execution cycles must be measurable (this is what finally feeds the
perf counter, `idle_cycles`, and `memory_bandwidth_gbs`).

## What software needs back

1. Achieved clock frequency (bandwidth/stall model input).
2. DMA overlap semantics (decides whether hoisted LOADs earn cycle credit).
3. Perf-counter behavior (`PERF_CYCLES`/`PERF_MACS` semantics).
4. Register signoff + transport choice (steps 4–5).

Until these arrive, `idle_cycles` stays 0, `memory_bandwidth_gbs`
stays 0.0, and `enpu_read_cycles` stays a stub — reported, not
invented.
