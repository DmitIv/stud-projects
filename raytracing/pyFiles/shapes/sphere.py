import numpy as np
from typing import Tuple
from pyFiles.shapes.interface import IShape
from pyFiles.shapes.material import Material


class Sphere(IShape):
    def __init__(self, center: np.ndarray, radius: float, m: Material):
        super().__init__(m, center)
        self._r: float = radius

    def intersect(self, orig: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        l: np.ndarray = self._c - orig
        tca: float = np.vdot(l, direction)
        d2: float = np.vdot(l, l) - tca * tca

        if d2 > (self._r * self._r):
            return False, -1.
        
        thc: float = np.sqrt(self._r * self._r - d2)
        t0: float = tca - thc
        t1: float = tca + thc

        if t0 > t1:
            t0, t1 = t1, t0

        if t0 < 0:
            t0 = t1
            if t0 < 0:
                return False, -1.
        return True, t0


