import sys
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVarTuple,
    runtime_checkable,
)

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

# Dimension TypeVars & Variadic Shape ---
N = TypeVar("N", default=Any)
M = TypeVar("M", default=Any)
K = TypeVar("K", default=Any)

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


# Underlying structural protocol for DLPack objects ---
@runtime_checkable
class _DLPackArray(Protocol):
    """Structural DLPack protocol satisfied by torch.Tensor, jax.Array, cupy.ndarray, etc."""

    def __dlpack__(
        self,
        *,
        stream: Any = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[Any, int] | None = None,
        copy: bool | None = None,
    ) -> Any: ...

    def __dlpack_device__(self) -> tuple[int, int]: ...


# Phantom parameters for rich IDE annotations ---
S = TypeVar("S")
D = TypeVar("D")
Dev = TypeVar("Dev")

# Generic Type Alias mapping phantom descriptors to the structural protocol
type Array[S, D, Dev] = _DLPackArray
