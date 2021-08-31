from fastai.vision import TfmPixel

from data_utils.data_environment import set_data_environment
from data_utils.data_loader import (
    SegItemListCustom, ImageListVertical, SegmentationItemListTwoSourceFoldersProb,
    SegmentationItemListSplitBatch
)
from data_utils.utility import _change_void_val
from data_utils.datasets import SeveralSourceDataset


def change_void_val(): return [TfmPixel(_change_void_val)()]


def change_void_val_tv():
    tmp = change_void_val()
    return [tmp, tmp]


__all__ = [
    set_data_environment,
    change_void_val,
    change_void_val_tv,
    SegItemListCustom,
    ImageListVertical,
    SegmentationItemListTwoSourceFoldersProb,
    SegmentationItemListSplitBatch,
    SeveralSourceDataset
]
