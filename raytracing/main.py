from render import scene_render
import numpy as np
import math
import time


RED_COLOR = np.array([255, 0, 0], dtype=np.uint8)
GREEN_COLOR = np.array([0, 255, 0], dtype=np.uint8)
BLUE_COLOR = np.array([0, 0, 255], dtype=np.uint8)
FIRST_LIGHT_SOURCE = np.array([250, -100, 0, 0.9], dtype=float)
SECOND_LIGHT_SOURCE = np.array([0, 600, 150, 1], dtype=float)
DEFAULT_SCENE = [
    1024,  # scene width
    768,  # scene height
    # x, y, z
    np.array([300, 250, 1000], dtype=float),  # eye position
    np.rad2deg(math.pi / 2.),  # fov in degrees
    np.array([[700, -250, 50, 50],
              [125, -340, 0, 50],
              [650, 500, -50, 150],
              ], dtype=float),  # spheres centers coords and radiuses
    [[RED_COLOR, 0.6, 0.4, 0.0, 50.],
     [GREEN_COLOR, 0.9, 0.1, 0.0, 50.],
     [BLUE_COLOR, 0.3, 0.3, 0.9, 2000.],
     ],  # materials conf
    #  count of spheres and materials must be equal
    [FIRST_LIGHT_SOURCE,
     SECOND_LIGHT_SOURCE],  # lights sources
    "full_thnd_reflecting_wall_render.png"  # output image name
]
WITH_REFLECTION = True
WITH_FLARE = True


if __name__ == "__main__":
    start_time: float = time.time()
    scene_render(shapes=DEFAULT_SCENE[4], materials=DEFAULT_SCENE[5],
                 lights=DEFAULT_SCENE[6], eye_position=DEFAULT_SCENE[2],
                 scene_width=DEFAULT_SCENE[0], scene_height=DEFAULT_SCENE[1],
                 fov_degree=DEFAULT_SCENE[3], output_image_name=DEFAULT_SCENE[7],
                 withReflection=WITH_REFLECTION, withFlare=WITH_FLARE)
    print(f"Elapsed time: {round(time.time() - start_time, ndigits=4):>4}")
