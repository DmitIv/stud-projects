import numpy as np
import cv2
import os
from typing import Union


class Player(object):
    def _save_after(self, new_image):
        self._images.append(new_image)

    def _save_now(self, new_image):
        cv2.imwrite(os.path.join(self._output_path, "image_{}.jpg".format(self._image_number)), new_image)
        self._image_number += 1

    def __init__(self, name: str, mode: str = 'save_after', output_path: str = './'):
        modes = ['save_after', 'save_now']
        handlers = [self._save_after, self._save_now]
        self._handlers_dict = dict(zip(modes, handlers))

        self._mode = mode
        if self._mode not in modes:
            self._mode = modes[0]

        self._name = name
        self._images = list()
        self._image_number = 0
        self._output_path = output_path

    def __add__(self, other: Union[np.ndarray, list]):
        if isinstance(other, list):

            if len(other) > 1:
                print('[WARNING]::Several detections of player on one frame.')
                for instance in other:
                    self._handlers_dict[self._mode](instance)
            elif len(other) == 1:
                self._handlers_dict[self._mode](other[0])

            else:
                return self

        else:
            self._handlers_dict[self._mode](other)

        return self

    def save_stored_images(self):
        images_count = 0
        for image in self._images:
            self._save_now(image)
            images_count += 1
        del self._images
        self._images = list()
        return images_count

    def get_images(self):
        return self._images


class NoSuchPlayerInDB(FileExistsError):
    def __init__(self, name: str, fullpath: str):
        FileExistsError()
        print("Player {0} doesn't exists. Full path : {1}".format(name, fullpath))


class WrongAddingPair(TypeError):
    def __init__(self, other_type):
        TypeError()
        print('Type of other object is {}, but should be Players or tuple<str, ndarray>.'.format(other_type))


class Players(object):
    def __init__(self, names: list, output_dir: str, default_mode: str):
        players = list()

        for name in names:
            player_output_path = os.path.join(output_dir, name)

            if not os.path.exists(player_output_path):
                raise NoSuchPlayerInDB(name, player_output_path)

            players.append(Player(name, default_mode, player_output_path))

        self._players_dict = dict(zip(names, players))

    def __add__(self, other):
        if other is None:
            return self

        elif isinstance(other, Players):
            for other_name, other_player in other._players_dict.items():
                self._players_dict[other_name] = self._players_dict[other_name] + other_player.get_images()
            del other

        elif isinstance(other, tuple):
            self._players_dict[tuple[0]] = self._players_dict[tuple[0]] + tuple[1]

        else:
            raise WrongAddingPair(type(other))

        return self

