import numpy as np

from pyFiles.scene.sceneFuncs import create_scene, show_scene
from pyFiles.render.renderingFuncs import (gradient_filling, simple_sphere_rendering, simple_shapes_rendering,
                                           shapes_rendering, shapes_rendering_phong, shapes_rendering_reflection)
from pyFiles.shapes.sphere import Sphere
from pyFiles.shapes.material import Material
from pyFiles.shapes.shapesContainer import ShapesContainer
from pyFiles.scene.light import Light
from typing import NoReturn, Union, List


__all__ = [
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6"
]


def stage1(scene_width: int, scene_height: int) -> NoReturn:
    scene: np.ndarray = create_scene(width=scene_width,
                                     height=scene_height)
    gradient_filling(scene)
    show_scene(scene, "Stage 1")
    del scene


def stage2(scene_width: int, scene_height: int,
           sphere_center: np.ndarray, sphere_radius: float,
           eye_position: np.ndarray,
           fov_degree: float,
           sphere_color: np.ndarray, background_color: np.ndarray) -> NoReturn:

    scene: np.ndarray = create_scene(width=scene_width,
                                     height=scene_height)

    sphere: Sphere = Sphere(center=sphere_center,
                            radius=sphere_radius,
                            m=Material(np.array([0, 0, 0])))

    simple_sphere_rendering(scene=scene, sphere=sphere,
                            fov_degree=fov_degree, eye_position=eye_position,
                            sphere_color=sphere_color,
                            background_color=background_color)

    show_scene(scene=scene, window_title="Stage 2")

    del scene


def stage3(scene_width: int, scene_height: int,
           spheres_centers: np.ndarray, spheres_radiuses: list,
           eye_position: np.ndarray,
           fov_degree: float,
           sphere_color: np.ndarray, background_color: np.ndarray) -> NoReturn:
    scene: np.ndarray = create_scene(width=scene_width,
                                     height=scene_height)

    shapes: ShapesContainer = ShapesContainer(1000)
    for i in range(0, len(spheres_radiuses)):
        shapes.append(Sphere(center=spheres_centers[i], radius=spheres_radiuses[i], m=Material(np.array([0, 0, 0]))))

    simple_shapes_rendering(scene=scene, shapes=shapes,
                            fov_degree=fov_degree, eye_position=eye_position,
                            sphere_color=sphere_color,
                            background_color=background_color)

    show_scene(scene=scene, window_title="Stage 3")

    del scene


def stage4(scene_width: int, scene_height: int,
           spheres_centers: np.ndarray, spheres_radiuses: list,
           colors: np.ndarray,
           eye_position: np.ndarray,
           fov_degree: float,
           light_sources: np.ndarray) -> NoReturn:
    scene: np.ndarray = create_scene(width=scene_width,
                                     height=scene_height)

    shapes: ShapesContainer = ShapesContainer(1000)
    for i in range(0, len(spheres_radiuses)):
        shapes.append(Sphere(center=spheres_centers[i], radius=spheres_radiuses[i], m=Material(colors[i])))

    lights: List[Light] = []
    for light_source in light_sources:
        lights.append(Light(position=light_source[:3], intensity=light_source[3]))

    shapes_rendering(scene=scene, shapes=shapes, fov_degree=fov_degree, eye_position=eye_position, lights=lights)

    show_scene(scene=scene, window_title="Stage 4")

    del scene


def stage5(scene_width: int, scene_height: int,
           spheres_centers: np.ndarray, spheres_radiuses: list,
           colors: np.ndarray, albedos: np.ndarray, spec_exponents: List[float],
           eye_position: np.ndarray,
           fov_degree: float,
           light_sources: np.ndarray) -> NoReturn:
    scene: np.ndarray = create_scene(width=scene_width,
                                     height=scene_height)

    shapes: ShapesContainer = ShapesContainer(1000)
    for i in range(0, len(spheres_radiuses)):
        shapes.append(Sphere(center=spheres_centers[i], radius=spheres_radiuses[i], m=Material(colors[i],
                                                                                               albedos[i],
                                                                                               spec_exponents[i])))

    lights: List[Light] = []
    for light_source in light_sources:
        lights.append(Light(position=light_source[:3], intensity=light_source[3]))

    shapes_rendering_phong(scene=scene, shapes=shapes, fov_degree=fov_degree, eye_position=eye_position, lights=lights)

    show_scene(scene=scene, window_title="Stage 5")

    del scene


def stage6(scene_width: int, scene_height: int,
           spheres_centers: np.ndarray, spheres_radiuses: list,
           colors: np.ndarray, albedos: np.ndarray, spec_exponents: List[float],
           eye_position: np.ndarray,
           fov_degree: float,
           light_sources: np.ndarray,
           depth_limit: int) -> NoReturn:
    scene: np.ndarray = create_scene(width=scene_width,
                                     height=scene_height)

    shapes: ShapesContainer = ShapesContainer(10000)
    for i in range(0, len(spheres_radiuses)):
        shapes.append(Sphere(center=spheres_centers[i], radius=spheres_radiuses[i], m=Material(colors[i],
                                                                                               albedos[i],
                                                                                               spec_exponents[i])))

    lights: List[Light] = []
    for light_source in light_sources:
        lights.append(Light(position=light_source[:3], intensity=light_source[3]))

    shapes_rendering_reflection(scene=scene, shapes=shapes, fov_degree=fov_degree, eye_position=eye_position,
                                lights=lights, depth_limit=depth_limit)

    show_scene(scene=scene, window_title="Stage 6")

    del scene

