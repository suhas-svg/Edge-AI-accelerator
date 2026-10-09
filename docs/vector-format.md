# Vector format v0.1 (frozen contract, software and hardware)

Golden vectors live in `tests/vectors/`. They are the shared reference for
RTL verification (spec sections 22 and 23). Software writes them; hardware
reads them and writes results back into the same file.

## File naming

`matmul_<M>x<K>x<N>.npz` where the name records the input shapes.
`matmul_overflow.npz` is the fixed overflow case: every element of `a_q` is
-128 and every element of `b_q` is 127.

## Contents

| Key | Type | Written by | Meaning |
| --- | --- | --- | --- |
| `a_q` | int8 `(M, K)` | software | quantized left input |
| `b_q` | int8 `(K, N)` | software | quantized right input |
| `expected` | int32 `(M, N)` | software | golden INT32 result, `a_q @ b_q` |
| `rtl_out` | int32 `(M, N)` | hardware | RTL simulation result, same shape |

`expected` is always the exact INT32 accumulation with no saturation. The
INT32 accumulator cannot overflow for 8x8 tiles.

## Who writes what

Software owns `a_q`, `b_q`, and `expected` and regenerates them with
`tools/gen_vectors.py`. Hardware never edits those keys.

After an RTL run, hardware adds the `rtl_out` key. A vector without
`rtl_out` counts as a failure, not a skip, so a partial run cannot pass by
omission.

## Verification

`.venv/Scripts/python.exe tools/verify_rtl.py tests/vectors/` compares
`rtl_out` against `expected` per vector and exits 1 on any mismatch,
shape mismatch, or dtype mismatch. It reports the first differing index
with both values.

`scripts/check.sh` runs the harness against a deliberately corrupted copy
via `tools/selfcheck.py`, proving the harness still fails on a bad result.

## Changing this contract

Any change to the keys, the naming, or the tolerance needs the doc updated,
`tools/gen_vectors.py` updated, and `tests/test_gen_vectors.py` plus
`tests/test_verify_rtl.py` updated in the same commit.
