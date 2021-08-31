import math

import numpy as np

from pyFiles import stage6


def main():
    scene_width = 1024
    scene_height = 768

    # stage1(scene_width, scene_height)

    # x, y, z
    sphere_center = np.array([42, 35, 0])
    sphere_radius = 125

    eye_position = np.array([300, 250, 1000])
    fov_degree = np.rad2deg(math.pi / 2.)

    sphere_color = np.array([255, 0, 0])
    background_color = np.array([255, 255, 255])

    # stage2(scene_width, scene_height, sphere_center, sphere_radius, eye_position, fov_degree,
    #        sphere_color, background_color)

    spheres_centers = np.array([
        [700, -250, 50],
        [125, -340, 0],
        [650, 500, -50],
    ])
    spheres_radiuses = [50, 50, 150]  # , 50, 150]
    # spheres_radiuses = []
    # stage3(scene_width, scene_height, spheres_centers, spheres_radiuses, eye_position, fov_degree,
    #        sphere_color, background_color)

    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255]
    ], dtype=np.uint8)
    light_sources = np.array([
        [250, -100, 0, 1.7],
        [0, 600, 150, 1.5]
    ], dtype=int)
    # stage4(scene_width, scene_height, spheres_centers, spheres_radiuses, colors, eye_position, fov_degree,
    #        light_sources)

    albedos = np.array([
        [0.6, 0.4, 0.0],
        [0.9, 0.1, 0.0],
        [0.3, 0.3, 0.9]
    ])
    spec_exponents = [50., 50., 2000]  # , 10., 2000.]
    # spec_exponents = []
    # stage5(scene_width, scene_height, spheres_centers, spheres_radiuses, colors, albedos, spec_exponents, eye_position,
    #        fov_degree, light_sources)

    depth_limit = 3

    stage6(scene_width, scene_height, spheres_centers, spheres_radiuses, colors, albedos, spec_exponents, eye_position,
           fov_degree, light_sources, depth_limit)
    # stage6(scene_width, scene_height, [], [], [], [], [], eye_position,
    #        fov_degree, light_sources, depth_limit)


if __name__ == "__main__":
    main()
