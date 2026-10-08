# SDD ledger — plan: docs/superpowers/plans/2026-10-08-phase2-verification-harness.md

Pre-flight: Task 1 produces SIZES/matmul_*x*.npz; Task 2 consumes tests/vectors/*.npz keys expected/rtl_out. Sizes naming changes from matmul_8x8 to matmul_8x8x8 — Task 2 glob matmul_*.npz covers both. tests/test_simulator.py currently globs matmul_*x*.npz and asserts count==4; will break on new vector count. Ruling needed.
Task 1: complete (commits 5c55787..44f7c49, tests: bash scripts/check.sh -> 15 passed)
Task 1: Ruling: simulator parity set is square vectors only; gen_vectors emits 7 total; test_simulator globs matmul_*.npz and keeps files whose three dims are equal. Cost if wrong: one test edit.
Task 2: complete (commits 44f7c49..92735f6, tests: bash scripts/check.sh -> 22 passed + self-check FAIL as expected)
Task 1: Ruling (final): overflow test asserts 64 rows of -1040384, plan said one row; 64x64 matmul has 64 outputs.
Task 2: Ruling: np.argwhere yields np.int64, cast to int for readable messages; test fixture named matmul_2x2x2.npz so main glob finds it.
