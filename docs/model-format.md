# Model binary v0.1

Order: header (magic `0x454D`, version, tensor count, command-bytes length, all LE), tensor table, weights blob, node table, command stream.
Node table: node count u32, then per node op-len u8 + op bytes, input-count u8, per input name-len u8 + name bytes, output name-len u8 + name bytes.
Writer and reader live in `compiler/binary.py`. Quantization params per tensor (scale f32, zero point i8) ride in the tensor table; currently written as scale 1.0 / zp 0.
Version field rejects unknown versions at parse time. No silent fallback.
