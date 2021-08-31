import numpy as np
from typing import NoReturn, Union, List
from pyFiles.shapes.shapesContainer import ShapesContainer
from pyFiles.shapes.sphere import Sphere
from pyFiles.render.utils import (random_color, cast_ray, vector_normalize, cast_ray_sphere,
                                  cast_ray_material, cast_ray_phong, cast_ray_phong_recur)
from pyFiles.scene.light import Light


def gradient_filling(scene: np.ndarray) -> NoReturn:
    """

    :param scene:
    :return:
    """
    h: int = scene.shape[0]
    w: int = scene.shape[1]
    for i in range(0, h):
        for j in range(0, w):
            scene[i, j] = np.array([round((i / float(h)) * 255),
                                    round((j / float(w)) * 255),
                                    0],
                                   dtype=np.uint8)


def simple_sphere_rendering(scene: np.ndarray, sphere: Sphere, fov_degree: float,
                            eye_position: Union[np.ndarray, None] = None,
                            sphere_color: Union[np.ndarray, None] = None,
                            background_color: Union[np.ndarray, None] = None) -> NoReturn:
    """

    :param scene:
    :param sphere:
    :param fov_degree:
    :param eye_position:
    :param sphere_color:
    :param background_color:
    :return:
    """
    if sphere_color is None:
        sphere_color = random_color()
    if background_color is None:
        background_color = random_color()

    h: int = scene.shape[0]
    w: int = scene.shape[1]

    if eye_position is None:
        eye_position = np.array([0, 0, 0])

    fov: float = np.deg2rad(fov_degree)
    scale: float = np.tan(fov * 0.5)
    imageAspectRatio: float = float(w) / float(h)
    for i in range(0, h):
        for j in range(0, w):
            x: float = (2 * (i + 0.5) / float(w) - 1) * scale
            y: float = (1 - 2 * (j + 0.5) / float(h)) * scale * (1. / imageAspectRatio)

            direction: np.ndarray = vector_normalize(np.array([y, x, -1.], dtype=float))
            scene[i, j] = cast_ray_sphere(eye_position, direction, sphere, sphere_color, background_color)


def simple_shapes_rendering(scene: np.ndarray, shapes: ShapesContainer, fov_degree: float,
                            eye_position: Union[np.ndarray, None] = None,
                            sphere_color: Union[np.ndarray, None] = None,
                            background_color: Union[np.ndarray, None] = None) -> NoReturn:
    """

    :param scene:
    :param shapes:
    :param fov_degree:
    :param eye_position:
    :param sphere_color:
    :param background_color:
    :return:
    """
    if sphere_color is None:
        sphere_color = random_color()
    if background_color is None:
        background_color = random_color()

    h: int = scene.shape[0]
    w: int = scene.shape[1]

    if eye_position is None:
        eye_position = np.array([0, 0, 0])

    fov: float = np.deg2rad(fov_degree)
    scale: float = np.tan(fov * 0.5)
    imageAspectRatio: float = float(w) / float(h)
    for i in range(0, h):
        for j in range(0, w):
            x: float = (2 * (i + 0.5) / float(w) - 1) * scale
            y: float = (1 - 2 * (j + 0.5) / float(h)) * scale * (1. / imageAspectRatio)

            direction: np.ndarray = vector_normalize(np.array([y, x, -1.], dtype=float))
            scene[i, j] = cast_ray(eye_position, direction, shapes, sphere_color, background_color)


def shapes_rendering(scene: np.ndarray, shapes: ShapesContainer, fov_degree: float,
                     lights: List[Light], eye_position: Union[np.ndarray, None] = None) -> NoReturn:
    """

    :param scene:
    :param shapes:
    :param fov_degree:
    :param lights:
    :param eye_position:
    :return:
    """
    h: int = scene.shape[0]
    w: int = scene.shape[1]

    if eye_position is None:
        eye_position = np.array([0, 0, 0])

    fov: float = np.deg2rad(fov_degree)
    scale: float = np.tan(fov * 0.5)
    imageAspectRatio: float = float(w) / float(h)
    for i in range(0, h):
        for j in range(0, w):
            x: float = (2 * (i + 0.5) / float(w) - 1) * scale
            y: float = (1 - 2 * (j + 0.5) / float(h)) * scale * (1. / imageAspectRatio)

            direction: np.ndarray = vector_normalize(np.array([y, x, -1.], dtype=float))
            scene[i, j] = cast_ray_material(eye_position, direction, shapes, lights)


def shapes_rendering_phong(scene: np.ndarray, shapes: ShapesContainer, fov_degree: float,
                           lights: List[Light], eye_position: Union[np.ndarray, None] = None) -> NoReturn:
    """

    :param scene:
    :param shapes:
    :param fov_degree:
    :param lights:
    :param eye_position:
    :return:
    """
    h: int = scene.shape[0]
    w: int = scene.shape[1]

    if eye_position is None:
        eye_position = np.array([0, 0, 0])

    fov: float = np.deg2rad(fov_degree)
    scale: float = np.tan(fov * 0.5)
    imageAspectRatio: float = float(w) / float(h)
    for i in range(0, h):
        for j in range(0, w):
            x: float = (2 * (i + 0.5) / float(w) - 1) * scale
            y: float = (1 - 2 * (j + 0.5) / float(h)) * scale * (1. / imageAspectRatio)

            direction: np.ndarray = vector_normalize(np.array([y, x, -1.], dtype=float))
            scene[i, j] = cast_ray_phong(eye_position, direction, shapes, lights)


def shapes_rendering_reflection(scene: np.ndarray, shapes: ShapesContainer, fov_degree: float,
                                lights: List[Light], depth_limit: int,
                                eye_position: Union[np.ndarray, None] = None) -> NoReturn:
    """

    :param scene:
    :param shapes:
    :param fov_degree:
    :param lights:
    :param depth_limit:
    :param eye_position:
    :return:
    """
    h: int = scene.shape[0]
    w: int = scene.shape[1]

    if eye_position is None:
        eye_position = np.array([0, 0, 0])

    fov: float = np.deg2rad(fov_degree)
    scale: float = np.tan(fov * 0.5)
    imageAspectRatio: float = float(w) / float(h)
    for i in range(0, h):
        for j in range(0, w):
            x: float = (2 * (i + 0.5) / float(w) - 1) * scale
            y: float = (1 - 2 * (j + 0.5) / float(h)) * scale * (1. / imageAspectRatio)

            direction: np.ndarray = vector_normalize(np.array([y, x, -1.], dtype=float))
            scene[i, j] = cast_ray_phong_recur(eye_position, direction, shapes, lights, 0, depth_limit)

