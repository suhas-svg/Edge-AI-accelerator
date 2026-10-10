# EdgeNPU Architecture

Software-pipeline view as built. The hardware lane (RTL, DMA, FPGA) is the
partner's; everything below runs today against the simulator.

## Data path

```text
ONNX file
  │  compiler/frontend.py — parse, shape inference, op validation,
  │                         NCHW batch-1 squeeze to CHW
  ▼
FP32 Graph + weights + specs
  │  compiler/quantize.py — symmetric per-tensor INT8, calibration scales
  ▼
INT8 Graph + weights + specs
  │  compiler/middleend.py — schedule, optimize, legalize (pad-and-trim),
  │                           validate, memory plan, then lower
  │                           to the command stream
  ▼
model.bin
  │  compiler/binary.py — pack model.bin (format v0.1)
  │  python/edge_npu/sdk.py — load, chain execution, stats
  │  simulator/hardware_model.py — numerics + cycle model
  ▼
INT8 result + cycles
```

The C runtime (`runtime/`) sits alongside: header validation, buffers,
checked execution, error codes. It does not yet submit to hardware.

## Op coverage

| Version | Ops | Status |
|---|---|---|
| v0.1 | LOAD, MATMUL, STORE | done |
| v0.2 | RELU, BIAS_ADD | done, chain execution |
| v0.3 | CONV2D, MAX_POOL | done, conv-led chains |
| v0.4 | REQUANTIZE | done, scale in stream |

Command encoding is at version v0.4 (`docs/command-format.md`).
Model binary is at version v0.1 (`docs/model-format.md`).

## Numerical contract

- Inputs and weights: INT8, range [-128, 127].
- Accumulators and intermediate buffers: INT32.
- Final outputs: INT8 after REQUANTIZE.
- Simulator numerics are computed by code independent of the reference
  implementation, so parity tests are real cross-checks.

## Cycle model

One MAC per slot per cycle on an 8x8 array: cycles = total MACs / 64.
Elementwise, pool, and requantize steps cost zero modeled cycles.
These are computed ideals, not measured silicon; see the demo discussion
of computed vs measured.

## NCHW boundary

ONNX vision tensors arrive NCHW. The frontend squeezes batch N=1 to the
CHW interior the NPU uses. Batch > 1 is rejected. Weights keep their
(K, C, KH, KW) rank.

## Test that pins it

`bash scripts/check.sh` — full suite, RTL-vector harness self-check,
C asserts, and the end-to-end demo (`demo/demo_inference.py`).
