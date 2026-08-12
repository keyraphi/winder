from typing import Any, Generic, Literal, Protocol, TypeVar, TypeVarTuple, runtime_checkable

# Dimension TypeVars & Variadic Shape ---
N = TypeVar("N")
M = TypeVar("M")
K = TypeVar("K")

ShapeArgs = TypeVarTuple("ShapeArgs")


class Shape(Generic[*ShapeArgs]):
    """Shape descriptor supporting symbolic expressions like Shape[N, Literal[3]] or Shape[M]."""

    ...


# DType Sentinels ---
class float32: ...
class float64: ...
class uint32: ...
class int32: ...


# Device Sentinels ---
class cuda: ...
class cpu: ...


# Generic Phantom Parameters ---
S = TypeVar("S")
D = TypeVar("D")
Dev = TypeVar("Dev")


@runtime_checkable
class Array(Protocol[S, D, Dev]):
    """Framework-agnostic array protocol for any DLPack-compatible tensor.

    Accepts torch.Tensor, jax.Array, cupy.ndarray, or any object implementing
    the __dlpack__ specification while maintaining rich shape and dtype documentation in IDEs.
    """

    def __dlpack__(self, stream: Any = None) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
