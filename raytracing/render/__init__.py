from render.funcs.rendering import rendering_per_pixel, REFLECTION_MULT, FLARE_MULT, DEPTH_LIMIT
from pyFiles.scene.sceneFuncs import create_scene
import os
import numpy as np
import cv2


def with_reflection():
    global REFLECTION_MULT
    REFLECTION_MULT = 1.


def without_reflection():
    global REFLECTION_MULT
    REFLECTION_MULT = 0.


def with_flare():
    global FLARE_MULT
    FLARE_MULT = 1.


def without_flare():
    global FLARE_MULT
    FLARE_MULT = 0.


def set_depth_limit(new_depth_limit: int):
    global DEPTH_LIMIT
    DEPTH_LIMIT = new_depth_limit


def scene_render(
        shapes: np.ndarray,
        materials: list,
        lights: list,
        eye_position: np.ndarray,
        scene_width: int, scene_height: int,
        fov_degree: float, output_image_name: str,
        withReflection: bool, withFlare: bool
):
    if not os.path.exists("logs"):
        os.mkdir("logs")

    if not os.path.exists("output"):
        os.mkdir("output")

    if withReflection:
        with_reflection()
    else:
        without_reflection()

    if withFlare:
        with_flare()
    else:
        without_flare()
    scene: np.ndarray = create_scene(scene_width, scene_height)
    scene = rendering_per_pixel(scene, shapes, fov_degree, materials, lights, eye_position)
    scene = cv2.cvtColor(scene.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("Scene render", scene)
    cv2.imwrite(os.path.join("output", output_image_name), scene)
    cv2.waitKey(0)


__all__ = [
    "scene_render",
]