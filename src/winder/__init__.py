from .winder_backend import WinderEngine as _Engine

class Winder:
    def __init__(self, points, faces=None, normals=None):
        # The user passes a torch.Tensor, jax.Array, etc.
        # nb::ndarray handles the __dlpack__ conversion automatically!
        if faces is not None:
            self._engine = _Engine(points, faces)
        else:
            self._engine = _Engine(points, normals)

    def compute(self, queries):
        return self._engine.compute(queries)

__all__ = ["Winder"]
