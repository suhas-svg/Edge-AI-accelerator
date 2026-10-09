# Tensor format v0.1

Fields: name, dtype in {fp32, int8, int32}, shape, scale, zero_point.
Scale is positive. INT8 range is [-128, 127]. Accumulators are INT32.

## Scale direction

`scale` means FP32 units per INT8 step, so a value is quantized by dividing
by it. This holds for `quantize_int8` and for every `TensorSpec`.

`requantize` is the exception because it goes the other way, INT32
accumulator to INT8. Its `scale` is output units per accumulator step, so it
multiplies. The two conventions are opposite under the same parameter name;
the difference is the direction of the conversion, not a bug. Any new
quantization helper must state which direction it takes in its docstring.
