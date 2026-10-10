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

The SDK executor keys buffers by stream order, not by address decoding:
LOAD sizes must equal input then weight bytes in order; each compute
command's shape fields and byte size must match the live buffer; the final
STORE must match the output bytes. Intermediate STOREs are permitted; only
the last one is checked against the result.

## Alignment discipline

- MATMUL dims m/n/k: each divisible by 8.
- CONV2D dims K/C/OH/OW: each divisible by 8.
- Elementwise/pool/requantize: total elements divisible by 64.
- The C runner (`enpu_run_matmul`) additionally requires buffers sized to
  at least m·k, k·n, and m·n·4 bytes respectively.

## Test that pins it

`tests/test_codegen.py` (layout values and every rejection),
`tests/test_sdk.py` (stream-vs-buffer checks), `runtime/tests/test_runtime.c`
(buffer sizing).
