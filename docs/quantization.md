# EdgeNPU Quantization

INT8 inference with INT32 accumulation throughout.

## Scheme

Symmetric per-tensor: `scale = max_abs / 127`, `zero_point = 0`.
Implemented in `compiler/quantize.py`; primitives in
`python/edge_npu/reference.py`.

## Scale direction

Two conventions share the parameter name `scale` (see
`docs/tensor-format.md`):

- `quantize_int8`: FP32 units per INT8 step — **divides**.
- `requantize`: output units per accumulator step — **multiplies**.

Getting this backwards is the classic bug; every new quantization helper
must state its direction in its docstring.

## Flow

1. Weights: `quantize_weights` maps each FP32 initializer to INT8 plus its
   scale. Non-float input is rejected. All-zero tensors take scale 1.0.
2. Activations: `quantize_activations` takes caller-supplied calibration
   samples per tensor and records the peak-abs scale. Empty sample lists
   are rejected.
3. Specs: `quantize_specs` stamps INT8 specs with scales and promotes
   `matmul`/`conv2d` outputs to INT32 accumulators. Any tensor without a
   scale is a named error.
4. Stream: `lower_requantize` packs the scale as f32 bits in the command's
   `reserved` field and `zero_point + 128` in `m`, so the executor needs no
   side channel.

## Error measurement

`quantization_error(fp32_out, deq_out)` is the max absolute error between
an FP32 result and its dequantized INT8 twin. The demo reports it per run
(typical tiny-matmul values ≈ 0.4–0.6); the benchmark suite does not gate
on it.

## Test that pins it

`tests/test_quantize.py` — exact scales, rejection paths, and the
frontend→quantize→lower bridge.
