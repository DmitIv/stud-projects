from pyFiles.shapes.interface import IShape
from pyFiles.shapes.material import Material
import numpy as np
from typing import List, NoReturn, Tuple
import sys


class ShapesContainer:
    def __init__(self, distance_border: float, *shapes: IShape):
        self._container: List[IShape] = []
        self._min_distance: float = float(sys.maxsize - 1)
        self._distance_border: float = distance_border
        for shape in shapes:
            self._container.append(shape)

    def _reinit(self) -> NoReturn:
        self._min_distance: float = float(sys.maxsize - 1)

    def intersect_any(self, orig: np.ndarray, direction: np.ndarray) -> Tuple[bool, Material, np.ndarray, np.ndarray]:
        self._reinit()
        material: Material = Material(np.zeros([0, 0, 0]), np.array([0.9, 0., 0.1]), 10.)
        normal: np.ndarray = np.zeros(shape=(3,), dtype=np.float)
        hit: np.ndarray = np.zeros(shape=(3,), dtype=np.float)

        for shape in self._container:
            is_intersected, current_distance = shape.intersect(orig, direction)

            if is_intersected and (current_distance < self._min_distance):
                self._min_distance = current_distance
                hit = orig + direction * current_distance
                normal = hit - shape.center()
                material = shape.material()

        wall_distance: float = float(sys.maxsize - 1)
        if np.abs(direction[0]) > 0.001:
            # right
            d: float = -1. * (orig[0] + 550) / direction[0]
            point: np.ndarray = orig + direction * d
            if -390 < point[1] < 650:
                if 1000 > point[2] > -250:
                    if 0 < d < self._min_distance:
                        wall_distance = d
                        hit = point
                        normal = np.array([1, 0, 0])
                        material = Material(np.array([15, 76, 129], dtype=np.uint8), np.array([0.9, 0.1, 0.]),
                                            10.)

            # left
            d: float = -1. * (orig[0] - 800) / direction[0]
            point: np.ndarray = orig + direction * d
            if -390 < point[1] < 650:
                if 1000 > point[2] > -250:
                    if 0 < d < self._min_distance:
                        wall_distance = d
                        hit = point
                        normal = np.array([-1, 0, 0])
                        material = Material(np.array([252, 102, 114], dtype=np.uint8), np.array([0.9, 0.1, 0.]),
                                            10.)

            # bottom
            d: float = -1. * (orig[1] - 650) / direction[1]
            point: np.ndarray = orig + direction * d
            if -550 < point[0] < 800:
                if 1000 > point[2] > -250:
                    if 0 < d < self._min_distance:
                        wall_distance = d
                        hit = point
                        normal = np.array([0, -1, 0])
                        material = Material(np.array([3, 153, 141], dtype=np.uint8), np.array([0.9, 0.1, 0.]),
                                            10.)

            # top
            d: float = -1. * (orig[1] + 390) / direction[1]
            point: np.ndarray = orig + direction * d
            if -550 < point[0] < 800:
                if 1000 > point[2] > -250:
                    if 0 < d < self._min_distance:
                        wall_distance = d
                        hit = point
                        normal = np.array([0, 1, 0])
                        material = Material(np.array([102, 2, 60], dtype=np.uint8), np.array([0.9, 0.1, 0.]),
                                            10.)

            # back
            d: float = -1. * (orig[2] + 250) / direction[2]
            point: np.ndarray = orig + direction * d
            if -550 < point[0] < 800:
                if -390 < point[1] < 650:
                    if 0 < d < self._min_distance:
                        wall_distance = d
                        hit = point
                        normal = np.array([0, 0, 1])
                        material = Material(np.array([223, 230, 234], dtype=np.uint8), np.array([0.4, 0.4, 0.9]),
                                            2000.)

        return min(self._min_distance, wall_distance) < self._distance_border, material, normal, hit

    def append(self, *shapes: IShape):
        for shape in shapes:
            self._container.append(shape)