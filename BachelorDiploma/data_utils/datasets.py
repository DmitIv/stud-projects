from pathlib import Path
from typing import Union, List, Tuple, Callable, Any

import fastai
from fastai.vision import (
    open_image, open_mask
)
from torch.utils.data import Dataset as t_Dataset


class SeveralSourceDataset(t_Dataset):
    _modes = [
        "absolute", "relative"
    ]

    def __init__(self, sources: Union[List[Union[Path, str]], Tuple[Union[Path, str]], Path, str],
                 batch_consistency: Union[List[int], Tuple[int]],
                 label_from_func: Callable, transforms: List[Any], size: int):
        t = type(sources)
        if t == Path or t == str:
            sources = [sources]

        self._sources: List[Path] = []
        for s in sources:
            if type(s) == str:
                self._sources.append(Path(s))
            else:
                self._sources.append(s)

        self._sources_count = len(self._sources)

        self._bc = batch_consistency

        if len(self._bc) != self._sources_count:
            raise RuntimeError("Unexpected consistency value")

        self._current_index = 0
        self._index_to_source = []
        index: int = 0
        for bc in self._bc:
            self._index_to_source += [index for _ in range(bc)]
            index += 1

        self.images: List[List[Path]] = []
        self.labels: List[List[Path]] = []

        self._len: int = 0
        index: int = 0
        for source in self._sources:
            self.images.append([])
            self.labels.append([])
            for image in source.iterdir():
                self.images[index].append(image)
                self.labels[index].append(label_from_func(image))
            self._len += len(self.images[index])
            index += 1

        self._inner_indexes: List[int] = [0 for _ in range(self._sources_count)]
        if type(transforms) != list:
            raise RuntimeError(
                "Unexpected transforms type. Should be list of fastai.vision.image.RandTransform or list of list of fastai.vision.image.RandTransform.")
        if len(transforms) > 0:
            if type(transforms[0]) == fastai.vision.image.RandTransform:
                transforms = [transforms]

        self.transform: List[List[Any]] = transforms
        self._size = size

    def __len__(self) -> int:
        return self._len

    def get(self, item):
        mod = len(self._index_to_source)
        index = self._index_to_source[item % mod]
        inner_index = self._inner_indexes[index]
        img = open_image(self.images[index][inner_index])
        label = open_mask(self.labels[index][inner_index], div=True)
        inner_index += 1
        self._inner_indexes[index] = inner_index % len(self.images[index])
        return img, label

    def __getitem__(self, item):
        img, lbl = self.get(item)
        img = img.resize(self._size).data
        lbl = lbl.resize(self._size).data.squeeze(dim=0)
        return img, lbl
