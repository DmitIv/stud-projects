from typing import List

import numpy as np

from pyFiles.scene.light import Light
from pyFiles.shapes.shapesContainer import ShapesContainer
from pyFiles.shapes.sphere import Sphere


def random_color() -> np.ndarray:
    np.random.seed(np.random.randint(1, 10000, dtype=np.int))
    r: np.uint8 = np.random.randint(0, 255, dtype=np.uint8)
    g: np.uint8 = np.random.randint(0, 255, dtype=np.uint8)
    b: np.uint8 = np.random.randint(0, 255, dtype=np.uint8)
    return np.array([r, g, b], dtype=np.uint8)


def cast_ray_sphere(orig: np.ndarray, direction: np.ndarray, sphere: Sphere,
                    sphere_color: np.ndarray, background_color: np.ndarray) -> np.ndarray:
    if sphere.intersect(orig, direction)[0]:
        return sphere_color
    else:
        return background_color


def cast_ray(orig: np.ndarray, direction: np.ndarray, shapes: ShapesContainer,
             sphere_color: np.ndarray, background_color: np.ndarray) -> np.ndarray:
    if shapes.intersect_any(orig, direction)[0]:
        return sphere_color
    else:
        return background_color


def cast_ray_material(orig: np.ndarray, direction: np.ndarray, shapes: ShapesContainer,
                      lights: List[Light]) -> np.ndarray:
    intersected_any, material, normal, point = shapes.intersect_any(orig, direction)
    if intersected_any:
        normal = vector_normalize(normal)
        light_intensity: float = 0
        for light in lights:
            light_dir: np.ndarray = vector_normalize(light.position() - point)
            scalar_product: float = max(0.0, np.vdot(light_dir, normal))
            light_intensity += light.intensity() * scalar_product
        return material.material() * light_intensity
    else:
        return np.array([0, 0, 0])


def cast_ray_phong(orig: np.ndarray, direction: np.ndarray, shapes: ShapesContainer,
                   lights: List[Light]) -> np.ndarray:
    intersected_any, material, normal, point = shapes.intersect_any(orig, direction)
    if intersected_any:
        normal = vector_normalize(normal)
        light_intensity: float = 0
        specular_light_intensity: float = 0
        for light in lights:
            light_dir: np.ndarray = vector_normalize(light.position() - point)
            scalar_product: float = max(0.0, np.vdot(light_dir, normal))
            light_intensity += light.intensity() * scalar_product
            specular_light_intensity += pow(max(0., np.vdot(reflect(vector_normalize(light_dir), normal), direction)),
                                            material.specular_exponent()) * light.intensity()
        return material.material() * light_intensity * material.albedo()[0] + \
               np.array([255., 255., 255.], dtype=np.float) * specular_light_intensity * material.albedo()[1]
    else:
        return np.array([0, 0, 0])


def cast_ray_phong_recur(orig: np.ndarray, direction: np.ndarray, shapes: ShapesContainer,
                   lights: List[Light], depth: int, depth_limit: int) -> np.ndarray:

    intersected_any, material, normal, point = shapes.intersect_any(orig, direction)

    if intersected_any and depth <= depth_limit:

        normal = vector_normalize(normal)
        light_intensity: float = 0
        specular_light_intensity: float = 0

        reflect_dir: np.ndarray = reflect(vector_normalize(direction), normal)
        reflect_orig: np.ndarray = point + normal * 0.001
        if np.vdot(reflect_dir, normal) < 0:
            reflect_orig = point - normal * 0.001
        reflect_color: np.ndarray = cast_ray_phong_recur(reflect_orig, reflect_dir, shapes, lights, depth + 1,
                                                         depth_limit)

        for light in lights:
            light_dir: np.ndarray = vector_normalize(light.position() - point)

            # TODO: Shadow debug
            light_distance: float = np.linalg.norm(light.position() - point)
            shadow_orig: np.ndarray = point + normal * 0.001
            if np.vdot(light_dir, normal) < 0:
                shadow_orig = point - normal * 0.001
            shadow_intersect, _, _, shadow_point = shapes.intersect_any(shadow_orig, light_dir)
            if shadow_intersect and np.linalg.norm(shadow_point - shadow_orig) < light_distance:
                continue

            scalar_product: float = max(0.0, np.vdot(light_dir, normal))
            light_intensity += light.intensity() * scalar_product
            specular_light_intensity += pow(max(0., np.vdot(reflect(vector_normalize(light_dir), normal), direction)),
                                            material.specular_exponent()) * light.intensity()

        color1 = material.material() * light_intensity * material.albedo()[0]
        color2 = np.array([255., 255., 255.], dtype=np.float) * specular_light_intensity * material.albedo()[1]
        color3 = reflect_color * material.albedo()[2]
        return np.clip(color1 + color2 + color3, 0, 255).astype(np.uint8)

    else:
        return np.array([0, 0, 0])


def vector_normalize(vector: np.ndarray) -> np.ndarray:
    norm: float = np.linalg.norm(vector)
    if norm == 0:
        norm = 1.
    return vector / norm


def reflect(i: np.ndarray, n: np.ndarray) -> np.ndarray:
    return i - n * 2. * np.vdot(i, n)
