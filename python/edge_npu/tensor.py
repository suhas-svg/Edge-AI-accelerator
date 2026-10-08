"""Tensor format. Parse once at the edge, trust the interior."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

DType = Literal["fp32", "int8", "int32"]
OpKind = Literal["matmul", "relu", "bias_add", "conv2d", "max_pool", "requantize", "load", "store"]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: DType
    shape: tuple[int, ...]
    scale: float = 1.0
    zero_point: int = 0

    def __post_init__(self) -> None:
        assert self.name, "tensor needs a name"
        assert all(d > 0 for d in self.shape), f"bad shape {self.shape}"
        assert self.scale > 0, f"scale must be positive, got {self.scale}"
