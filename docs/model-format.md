# Model binary v0.1

Order: header (magic, version, counts), tensor table, weights blob, quantization params, command stream.
Version field rejects unknown versions at parse time. No silent fallback.
