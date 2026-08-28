from . import types

try:
    from .winder_module import (
        WindingNumberEngine,
        GradientEngine,
        brute_force_winding_numbers,
        brute_force_gradients,
    )
except ImportError as e:
    import warnings

    warnings.warn(
        f"Failed to load compiled extension 'winder_module': {e}. "
        "Native C++/CUDA routines will be unavailable.",
        ImportWarning,
        stacklevel=2,
    )

from . import torch

__all__ = [
    "WindingNumberEngine",
    "GradientEngine",
    "brute_force_winding_numbers",
    "brute_force_gradients",
    "types",
    "torch",
]
