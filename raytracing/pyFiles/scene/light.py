import numpy as np


class Light:
    def __init__(self, position: np.ndarray, intensity: float):
        self._p: np.ndarray = position
        self._ity: float = intensity

    def position(self) -> np.ndarray:
        return self._p

    def intensity(self) -> float:
        return self._ity
