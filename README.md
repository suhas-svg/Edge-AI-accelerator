# EdgeNPU - software workspace

Software side of the EdgeNPU project (INT8 edge-AI accelerator on FPGA).

## Roles

- Software (you): `compiler/`, `runtime/`, `python/`, `simulator/`, `benchmarks/`, `tests/`
- Hardware (partner): `rtl/`, `firmware/` low-level, FPGA bringup

Shared contracts live in `docs/`: register map, command format, tensor format, model format.
No code crosses those contracts without updating the doc and the test first.

## Quickstart (Windows, Git Bash)

```bash
bash scripts/setup.sh
bash scripts/check.sh
```

Setup is idempotent. Run it twice, it converges to the same state.
