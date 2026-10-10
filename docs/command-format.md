# Command format v0.2

Header: 6 bytes LE: magic 0x454E (u16) + count (u32).
Each command: 19 bytes LE: op u8, address u32, size u32, reserved u32, m u16, n u16, k u16.
Opcodes: LOAD=0x01, MATMUL=0x02, STORE=0x03, RELU=0x04, BIAS_ADD=0x05.
Elementwise ops (RELU, BIAS_ADD) use address + size for the target INT32 buffer in bytes; m/n/k are zero. v0.1 streams (LOAD/MATMUL/STORE only) decode unchanged under v0.2.

Python encode and decode live in `python/edge_npu/commands.py`.
