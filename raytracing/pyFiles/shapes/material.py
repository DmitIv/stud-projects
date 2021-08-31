import numpy as np
from typing import Union


class Material:
    def __init__(self, color: np.ndarray, a: Union[np.ndarray, None] = None, spec: Union[float, None] = None):
        self._color: np.ndarray = color
        self._a: np.ndarray = a
        if self._a is None:
            self._a = np.array([0., 0., 0.])
        self._spec: float = spec
        if self._spec is None:
            self._spec = 1.

    def material(self) -> np.ndarray:
        return self._color

    def albedo(self) -> np.ndarray:
        return self._a

    def specular_exponent(self) -> float:
        return self._spec
