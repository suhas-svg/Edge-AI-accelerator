# EdgeNPU Benchmarking

How the numbers in `benchmarks/results.csv` are produced and what they mean.

## How to run

```bash
bash scripts/setup.sh   # once; idempotent
.venv/Scripts/python.exe benchmarks/bench_matmul.py
```

The script rewrites `benchmarks/results.csv` and prints the rows. Commit the
regenerated file as the reference snapshot. It takes a few minutes (the 256
case dominates).

## Methodology

- Fixed seed (`SEED = 0`, per-row derived streams), so inputs are identical
  run to run. Only wall-clock medians vary.
- Each timing is the median of `REPEATS = 5` runs after `WARMUP = 2` untimed
  runs. A single shot catches OS jitter; the median does not.
- Wall-clock columns (`fp32_ms`, `int8_ms`, `edgenpu_sim_ms`) are reference
  snapshots for humans, never gates. No test asserts on them.

## Computed vs measured

`edgenpu_cycles` is computed, not measured: total MACs divided by the 64
slots of the 8x8 array, on padded tile geometry for odd dims. There is no
silicon clock behind it; treat it as an ideal, and see the demo discussion
of computed vs measured. `edgenpu_sim_ms` is measured host time running the
simulator, useful only as a rough cost comparison against the CPU columns.

## Columns

| Column | Meaning |
|---|---|
| `case` | workload label (`matmul-<m>x<k>x<n>`, `conv-chain-8ch`, `requantize-chain-64`) |
| `fp32_ms` | CPU FP32 reference, blank where no FP32 reference exists for that op |
| `int8_ms` | CPU INT8 reference for the whole row |
| `edgenpu_sim_ms` | full compile→pack→load→infer path through the SDK |
| `edgenpu_cycles` | computed ideal cycles (equals `busy_cycles`) |
| `busy_cycles` | MAC-array busy cycles from `Model.get_stats()` |
| `dma_bytes` | LOAD plus STORE bytes from `Model.get_stats()` |
| `command_counts` | per-opcode tally, stream order (`LOAD:2;MATMUL:1;STORE:1`) |

## Test that pins it

`tests/test_bench.py` builds tiny rows through the same row functions and
asserts exact cycles, DMA bytes, and count strings. The full bench itself
stays out of `scripts/check.sh` (minutes, not seconds).
