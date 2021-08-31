import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
import Demo1.PlayersClasses as pC
from InsightFace_Pytorch.mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from typing import Union


class Frame(object):
    def __init__(self, view: np.ndarray, frame_number: int):
        self._frame_number = frame_number
        self._view = view

    def modify_view(self, handler: callable, **kwargs):
        self._view = handler(self._view, **kwargs)

    def get_view(self):
        return self._view

    def __str__(self):
        return self._frame_number


class View(ABC):
    @abstractmethod
    def analysis(self) -> Union[pC.Players, None]:
        pass


class ZoomViewWrongType(TypeError):
    def __init__(self, frame_number: int):
        TypeError()
        print("Frame number {}. Wrong instance of detector or matcher.".format(frame_number))


class ZoomView(View):
    def __init__(self, view: np.ndarray, frame_number: int, face_detector: any, face_matcher: any,
                 names: any, targets: any):
        if not hasattr(face_detector, 'detect') or not hasattr(face_matcher, 'match'):
            raise ZoomViewWrongType(frame_number)
        self._frame = Frame(view, frame_number)
        self._detector = face_detector
        self._matcher = face_matcher
        self._names = names
        self._targets = targets
        self._bboxes = None

    def _convert_for_matcher(self, result_of_detector: list):
        bboxes = []
        faces = []

        for face_annotation in result_of_detector:
            bbox = face_annotation[0][:4]
            landmarks = face_annotation[1]
            face = Image.fromarray(
                warp_and_crop_face(np.array(self._frame.get_view()), landmarks,
                                   get_reference_facial_points(default_square=True),
                                   crop_size=(112, 112)))
            bboxes.append(bbox)
            faces.append(face)

        return faces, np.vstack(bboxes)

    def analysis(self) -> Union[pC.Players, None]:

        result_of_detector = self._detector(self._frame.get_view())
        faces, bboxes = self._convert_for_matcher(result_of_detector)

        if len(faces) == 0:
            print('[WARNING]:: No face')
            return None
        else:
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
            self._bboxes = bboxes

            results, score = self._matcher.match(faces, self._targets)

            players_on_frame = pC.Players(self._names, './', 'save_after')
            new_names = []
            for idx, face in enumerate(faces):
                name = self._names[results[idx] + 1]
                players_on_frame = players_on_frame + (name, np.array(face))
                new_names.append(name)
            self._names = new_names

        return players_on_frame
