# Tensor format v0.1

Fields: name, dtype in {fp32, int8, int32}, shape, scale, zero_point.
Scale is positive. INT8 range is [-128, 127]. Accumulators are INT32.
