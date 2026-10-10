# EdgeNPU Runtime API

Two surfaces: C for the device layer, Python for developers. Both execute
against the simulator until FPGA exists.

## C API (`runtime/include/edge_npu.h`)

Lifecycle: `enpu_open` → `enpu_load_model` → alloc/run → `enpu_free_buffer`.

| Function | Behavior |
|---|---|
| `enpu_open(dev)` | marks device present; ARG on null |
| `enpu_load_model(dev, path, model)` | validates model.bin header magic + version; IO on file errors, VERSION on mismatch |
| `enpu_alloc_buffer(dev, size, buf)` | zeroed buffer; ARG on zero size or dead device, NOMEM on failure |
| `enpu_free_buffer(buf)` | null-safe, double-free-safe, clears the struct |
| `enpu_matmul_i8(a, b, c, m, n, k)` | raw INT8 kernel, INT32 out |
| `enpu_cycles_for_matmul(m, n, k, cycles)` | MACs/64; ARG unless dims are tile multiples |
| `enpu_run_matmul(dev, a, b, c, m, n, k, cycles)` | size-checked buffers + kernel + cycles in one call |
| `enpu_read_cycles(cycles)` | stub 0 until the perf counter exists |

Error codes: `ENPU_OK = 0`, `ENPU_ERR_ARG = -1`, `ENPU_ERR_NOMEM = -2`,
`ENPU_ERR_IO = -3`, `ENPU_ERR_VERSION = -4`.

## C++ API (`runtime/include/edge_npu_cpp.h`)

RAII over the C API, compiled from `runtime/src/edge_npu_cpp.cpp`
(`g++ -std=c++14`). `enpu::Device` opens on construction, `enpu::Model`
loads and validates on construction, `enpu::Buffer` frees on destruction
and is move-only. Every method throws `enpu::Error` (a `std::runtime_error`
carrying the `ENPU_*` code) on failure; `run_matmul` returns the cycle
count.

## Python SDK (`python/edge_npu/sdk.py`)

```python
device = Device()
model = device.load_model("model.bin")
result = model.predict(x)
stats = model.get_stats()   # cycles, mac_utilization, memory_bandwidth_gbs
bench = device.benchmark(model, x)  # median ms
```

- `load_model` parses with `read_model`; corrupt files raise `ValueError`.
- `predict(x)` executes chains starting with `matmul` or `conv2d`,
  followed by `relu`/`bias_add`/`max_pool`/`requantize`, with wiring
  checked at load and every command's shape/size checked at run.
  Anything else is a `ValueError`.
- `memory_bandwidth_gbs` is 0.0 until a clock assumption exists — reported,
  not invented.
- Elementwise/pool/requantize steps cost zero modeled cycles.
- Cycles count executed (padded) MACs; `predict` trims to the logical
  shape, so numerics match the logical reference bit-exact.

## Test that pins it

`runtime/tests/test_runtime.c` and `runtime/tests/test_runtime_cpp.cpp`
(both compiled and run by `scripts/check.sh`),
`tests/test_sdk.py` (chain execution, stats, every rejection).
