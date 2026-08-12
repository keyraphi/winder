from . import types

try:
    from .winder_module import *
except ImportError:
    # Extension module hasn't been copied/installed into this directory yet
    pass

__all__ = [
    "WindingNumberEngine",
    "GradientEngine",
    "brute_force_winding_numbers",
    "brute_force_gradients",
    "types",
]
