# Command format v0.4

Header: 6 bytes LE: magic 0x454E (u16) + count (u32).
Each command: 19 bytes LE: op u8, address u32, size u32, reserved u32, m u16, n u16, k u16.
Opcodes: LOAD=0x01, MATMUL=0x02, STORE=0x03, RELU=0x04, BIAS_ADD=0x05, CONV2D=0x06, MAX_POOL=0x07, REQUANTIZE=0x08.
Elementwise ops (RELU, BIAS_ADD) use address + size for the target INT32 buffer in bytes; m/n/k are zero. v0.1 streams (LOAD/MATMUL/STORE only) decode unchanged under v0.2.
CONV2D uses address + size for the output INT32 buffer in bytes; m=K output channels, n=OH, k=OW. Valid padding, stride 1 only; kernel and input channels resolve from weights at run. MAX_POOL uses address + size for the output buffer; m=window, n=stride.
REQUANTIZE uses address + size for the output INT8 buffer in bytes; reserved holds the scale as LE f32 bits; m holds zero_point + 128; n/k are zero.

Python encode and decode live in `python/edge_npu/commands.py`.
