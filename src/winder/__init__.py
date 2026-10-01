from . import types

try:
    from .winder_module import (
        WindingNumberEngine,
        GradientEngine,
        brute_force_winding_numbers_mesh,
        brute_force_winding_numbers_point_normal,
        brute_force_winding_numbers_triangle_soup,
        brute_force_gradients_mesh,
        brute_force_gradients_point_normal,
        brute_force_gradients_triangle_soup,
    )
except ImportError as e:
    import warnings

    warnings.warn(
        f"Failed to load compiled extension 'winder_module': {e}. "
        "Native C++/CUDA routines will be unavailable.",
        ImportWarning,
        stacklevel=2,
    )


__all__ = [
    "WindingNumberEngine",
    "GradientEngine",
    "brute_force_winding_numbers_mesh",
    "brute_force_winding_numbers_point_normal",
    "brute_force_winding_numbers_triangle_soup"
    "brute_force_gradients_mesh",
    "brute_force_gradients_point_normal",
    "brute_force_gradients_triangle_soup",
    "types",
]
