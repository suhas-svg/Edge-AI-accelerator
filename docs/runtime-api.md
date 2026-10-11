# EdgeNPU Runtime API

Two surfaces: C for the device layer, Python for developers. Both execute
against the simulator until FPGA exists.

## C API (`runtime/include/edge_npu.h`)

Lifecycle: `enpu_open` → `enpu_load_model` → alloc/run → `enpu_free_buffer`.
`enpu_reset` tears the device down (later calls fail ARG until `enpu_open`
runs again); already-allocated buffers and loaded model structs keep their
bytes.

| Function | Behavior |
|---|---|
| `enpu_open(dev)` | marks device present; ARG on null |
| `enpu_reset(dev)` | clears present; ARG on null or dead device |
| `enpu_load_model(dev, path, model)` | validates model.bin header magic + version; IO on file errors, VERSION on mismatch |
| `enpu_alloc_buffer(dev, size, buf)` | zeroed buffer; ARG on zero size or dead device, NOMEM on failure |
| `enpu_free_buffer(buf)` | null-safe, double-free-safe, clears the struct |
| `enpu_matmul_i8(a, b, c, m, n, k)` | raw INT8 kernel, INT32 out |
| `enpu_relu_i32(buf, n)` | in-place max(x, 0) on INT32; ARG on null/zero |
| `enpu_bias_add_i32(buf, bias, n)` | exact-count add; 1D broadcast stays Python-side |
| `enpu_requantize_i32(acc, out, n, scale, zp)` | round-half-even, clamp INT8; ARG on bad scale/zp |
| `enpu_max_pool_i32(x, y, c, h, w, size, stride)` | window max; ARG on oversized window |
| `enpu_conv2d_i8(x, w, y, c, h, wd, k, kh, kw)` | valid stride-1, square kernel, INT32 out |
| `enpu_cycles_for_matmul(m, n, k, cycles)` | MACs/64; ARG unless dims are tile multiples |
| `enpu_cycles_for_conv2d(k, oh, ow, c, kh, kw, cycles)` | MACs/64; ARG unless K/C/OH/OW are tile multiples |
| `enpu_run_matmul(dev, a, b, c, m, n, k, cycles)` | size-checked buffers + kernel + cycles in one call |
| `enpu_run_conv2d(dev, x, w, y, c, h, wd, k, kh, kw, cycles)` | size-checked conv + cycles; ARG unless K/C/OH/OW tile-aligned |
| `enpu_run_relu/bias_add/requantize/max_pool(...)` | size-checked elementwise/pool, zero cycles; elementwise needs numel divisible by 64 |
| `enpu_read_cycles(cycles)` | stub 0 until the perf counter exists |

Error codes: `ENPU_OK = 0`, `ENPU_ERR_ARG = -1`, `ENPU_ERR_NOMEM = -2`,
`ENPU_ERR_IO = -3`, `ENPU_ERR_VERSION = -4`.

## C++ API (`runtime/include/edge_npu_cpp.h`)

RAII over the C API, compiled from `runtime/src/edge_npu_cpp.cpp`
(`g++ -std=c++14`). `enpu::Device` opens on construction, `enpu::Model`
loads and validates on construction, `enpu::Buffer` frees on destruction
and is move-only. Every method throws `enpu::Error` (a `std::runtime_error`
carrying the `ENPU_*` code) on failure; `run_matmul` returns the cycle
count. `reset()` tears down (use throws until `open()` runs again).
`run_conv2d` and `cycles_for_conv2d` cover conv; `run_relu`,
`run_bias_add`, `run_requantize`, and `run_max_pool` cover the tails and
return the zero cycle count.

## Python SDK (`python/edge_npu/sdk.py`)

```python
device = Device()
model = device.load_model("model.bin")
result = model.predict(x)
stats = model.get_stats()   # cycles, mac_utilization, memory_bandwidth_gbs,
                            # busy_cycles, idle_cycles, dma_bytes, command_counts
bench = device.benchmark(model, x)  # median ms
```

- `load_model` parses with `read_model`; corrupt files raise `ValueError`.
- `predict(x)` executes chains starting with `matmul` or `conv2d`,
  followed by `relu`/`bias_add`/`max_pool`/`requantize`/`matmul`/`conv2d`,
  with wiring checked at load and every command's shape/size checked at run.
  Later computes stage source inputs from LOADs and reuse live buffers;
  cycles and MACs accumulate across computes. Anything else is a
  `ValueError`.
- `memory_bandwidth_gbs` is 0.0 until a clock assumption exists — reported,
  not invented.
- Elementwise/pool/requantize steps cost zero modeled cycles.
- `busy_cycles` is the sim cycle count; `idle_cycles` is 0 until a stall
  model exists. `dma_bytes` sums LOAD plus STORE sizes; `command_counts`
  tallies opcodes in the packed stream.
- Cycles count executed (padded) MACs; `predict` trims to the logical
  shape, so numerics match the logical reference bit-exact.
- `reset()` closes the device (`load_model`, `benchmark`, `info`,
  `get_stats` raise until `open()`); already-loaded models carry their
  own data and keep predicting.

## Test that pins it

`runtime/tests/test_runtime.c` and `runtime/tests/test_runtime_cpp.cpp`
(both compiled and run by `scripts/check.sh`),
`tests/test_sdk.py` (chain execution, stats, every rejection).
