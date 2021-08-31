from random import uniform

import torch as t_
from fastai.vision import (
    SegmentationItemList, SegmentationLabelList, open_mask, ImageList, open_image, DataBunch
)
from fastai.vision.transform import (
    dihedral
)
from torch import device as t_device
from torch.utils.data import Dataset


class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)


class SegItemListCustom(SegmentationItemList):
    _label_cls = SegLabelListCustom


class ImageListVertical(ImageList):
    def open(self, fn): return dihedral(open_image(fn), k=5)


class SegmentationItemListSourceBase:
    def __init__(self, source_num=2, **kwargs):
        if source_num == 0:
            raise RuntimeError("Incorrect number of data sources")
        self._sigILs: list = [SegItemListCustom for i in range(source_num)]
        self._segILsLabeled: list = [None for i in range(source_num)]

        self._split_by_folder = False
        self._transform = False
        self._ready = False
        self.c = 0

        self.path = None
        self.device = None

        self.loss_func = None

        self.train_ds = None
        self.train_dl = None

        self.val_ds = None
        self.val_dl = None

        self.test_ds = None
        self.test_dl = None

        self.batch_size = 0

    def get(self, i):
        raise NotImplementedError("Base class")

    def show_batch(self, **kwargs):
        raise NotImplementedError("Base class")

    def _data_ready(self):
        if not self._ready:
            raise RuntimeError("Data loader not ready yet")

    def reconstruct(self, t):
        self._data_ready()
        return self._segILsLabeled[0].reconstruct(t)

    def show_xys(self, xs, ys, figsize: tuple = (12, 6), **kwargs):
        self._data_ready()
        self._segILsLabeled[0].show_xys(xs, ys, figsize=figsize, **kwargs)

    def show_xyzs(self, xs, ys, zs, figsize: tuple = None, **kwargs):
        self._data_ready()
        self._segILsLabeled[0].show_xyzs(xs, ys, zs, figsize=figsize, **kwargs)

    def analyze_pred(self, pred):
        return self._segILsLabeled[0].analyze_pred(pred)

    def set_path_variable(self, path):
        self.path = path

    def set_device(self, device):
        self.device = device

    def from_folders(self, folders_list: list, **kwargs):
        if type(folders_list) != list and type(folders_list) != tuple:
            folders_list = [folders_list]
        mod = len(folders_list)
        for i in range(len(self._sigILs)):
            self._sigILs[i] = self._sigILs[i].from_folder(folders_list[i % mod], **kwargs)
        return self

    def split_by_folder(self, train_dirs, val_dirs):
        if type(train_dirs) != list and type(train_dirs) != tuple:
            train_dirs = [train_dirs]

        if type(val_dirs) != list and type(val_dirs) != tuple:
            val_dirs = [val_dirs]

        if len(train_dirs) != len(val_dirs):
            raise RuntimeError("Unexpected count of dirs")

        mod = len(train_dirs)
        for i in range(len(self._sigILs)):
            self._sigILs[i] = self._sigILs[i].split_by_folder(train=train_dirs[i % mod], valid=val_dirs[i % mod])

        self._split_by_folder = True
        return self

    def label_from_func(self, funcs, classes):
        if type(funcs) != list and type(funcs) != tuple:
            funcs = [funcs]

        if len(funcs) == 0:
            raise RuntimeError("Must be not less one function")

        mod = len(funcs)

        if self._split_by_folder:
            for i in range(len(self._sigILs)):
                self._segILsLabeled[i] = self._sigILs[i].label_from_func(func=funcs[i % mod], classes=classes)
        else:
            raise RuntimeError("Not been splited yet")
        return self

    def transform(self, tfms_per_source, size, tfm_y, **kwargs):
        if self._segILsLabeled[0] is None:
            raise RuntimeError("Not been labeled yet")

        if type(tfms_per_source) != list and type(tfms_per_source) != tuple:
            tfms_per_source = [tfms_per_source]
        if len(tfms_per_source) == 0:
            tfms_per_source = [None]
        mod = len(tfms_per_source)

        for i in range(len(self._segILsLabeled)):
            self._segILsLabeled[i] = self._segILsLabeled[i].transform(tfms=tfms_per_source[i % mod], tfm_y=tfm_y,
                                                                      size=size, **kwargs)

        self._transform = True

        return self

    def databunch_and_normalized(self, bs, stats):
        if type(bs) != list and type(bs) != tuple:
            bs = [bs]

        if self.device is None:
            if t_.cuda.is_available():
                self.device = t_device('cuda:0')
            else:
                self.device = t_device('cpu')

        mod_bs = len(bs)
        if self._transform:
            for i in range(len(self._segILsLabeled)):
                self._segILsLabeled[i] = self._segILsLabeled[i].databunch(bs=bs[i % mod_bs]).normalize(stats)
                self._segILsLabeled[i].device = self.device
            self.c = self._segILsLabeled[0].c

            for i in range(len(self._segILsLabeled)):
                if self._segILsLabeled[i].c != self.c:
                    raise RuntimeError("Miscasting classes num between sources")

            self._ready = True

            if self.path is None:
                self.path = self._segILsLabeled[0].path

            for b in bs:
                self.batch_size += b

            return self
        else:
            raise RuntimeError("Not been transformed yet")


class SegmentationItemListTwoSourceFoldersProb(SegmentationItemListSourceBase):
    def __init__(self, second_folder_probability: float = 0.9, **kwargs):
        super().__init__(source_num=2, **kwargs)
        if (second_folder_probability > 1.0) or (second_folder_probability < 0.0):
            raise RuntimeError("Not correct probability for second folder getting")
        self._second_folder_probability = second_folder_probability

    def _second_folder(self):
        return uniform(0.0, 1.0) > (1.0 - self._second_folder_probability)

    def get(self, i):
        self._data_ready()
        if self._second_folder():
            return self._segILsLabeled[1].get(i)
        return self._segILsLabeled[0].get(i)

    def show_batch(self, **kwargs):
        self._data_ready()
        if self._second_folder():
            self._segILsLabeled[1].show_batch(**kwargs)
        else:
            self._segILsLabeled[0].show_batch(**kwargs)


class SplitBatchDataset(Dataset):
    def __init__(self, datasets: list, batch_consistency: list):
        self._datasets = datasets
        self._len = len(self._datasets[0])
        for dataset in self._datasets:
            if len(dataset) > self._len:
                self._len = len(dataset)

        self._item_to_dataset = []
        index = 0
        self._common_batch_size = 0
        for bc in batch_consistency:
            self._item_to_dataset += [index for _ in range(bc)]
            self._common_batch_size += bc
            index += 1
        self._counter = 0

        self.x = self._datasets[0].x
        self.y = self._datasets[0].y

        print("X: ", type(self.x))
        print("Y: ", type(self.y))

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        dataset_index = self._item_to_dataset[self._counter % self._common_batch_size]
        self._counter += 1
        return self._datasets[dataset_index][item]


class SegmentationItemListSplitBatch(SegmentationItemListSourceBase):
    def __init__(self, batch_consistency: list, **kwargs):
        if type(batch_consistency) != list and type(batch_consistency) != tuple:
            batch_consistency = [batch_consistency]
        if len(batch_consistency) == 0:
            raise RuntimeError("Wrong batch consistency")
        super().__init__(source_num=len(batch_consistency), **kwargs)
        self._count = len(batch_consistency)
        self._batches = batch_consistency

    def get(self, i):
        self._data_ready()
        result = self._segILsLabeled[0].get(i)
        return result

    def show_batch(self, **kwargs):
        self._data_ready()
        for i in range(len(self._segILsLabeled)):
            self._segILsLabeled[i].show_batch(**kwargs)

    def databunch_and_normalized(self, bs, stats):
        final_cut = super().databunch_and_normalized(self._batches, stats)
        t_ds = SplitBatchDataset(
            datasets=[x.train_ds for x in self._segILsLabeled], batch_consistency=self._batches
        )
        v_ds = SplitBatchDataset(
            datasets=[x.valid_ds for x in self._segILsLabeled], batch_consistency=self._batches
        )
        return final_cut, DataBunch.create(
            train_ds=t_ds, valid_ds=v_ds, bs=self.batch_size
        )