from functools import partial

from fastai.metrics import (
    fbeta
)
from torch import float64 as f64


def accuracy_segmentation(inp, target):
    target = target.squeeze(1)
    return (inp.argmax(dim=1) == target).float().mean()


def jaccard_index(inp, target, base_value: int) -> float:
    target = target.squeeze(1)
    inp = inp.argmax(dim=1)
    intersection = ((inp == base_value) & (target == base_value)).unique(return_counts=True)
    union = ((inp == base_value) | (target == base_value)).unique(return_counts=True)
    return (intersection[1][1].to(f64) / union[1][1].to(f64)) * 100.0


def jaccard_index_part(inp, target, base_value: int, slice_start: int, slice_end: int = None):
    if slice_end is None:
        inp = inp[slice_start:]
        target = target[slice_start:]
    else:
        inp = inp[slice_start:slice_end]
        target = target[slice_start:slice_end]
    return jaccard_index(inp, target, base_value)


def jaccard_index_zero_class(inp, target) -> float:
    return jaccard_index(inp, target, 0)


def jaccard_index_one_class(inp, target) -> float:
    return jaccard_index(inp, target, 1)


def jaccard_index_one_class_first_part(inp, target, middle: int):
    return jaccard_index_part(inp, target, 1, 0, middle)


def jaccard_index_one_class_second_part(inp, target, middle: int):
    return jaccard_index_part(inp, target, 1, middle)


def jaccard_index_zero_class_first_part(inp, target, middle: int):
    return jaccard_index_part(inp, target, 0, 0, middle)


def jaccard_index_zero_class_second_part(inp, target, middle: int):
    return jaccard_index_part(inp, target, 0, middle)


def get_jaccard_index_one_class_partial(middle: int):
    return partial(jaccard_index_one_class_first_part, middle=middle), partial(jaccard_index_one_class_second_part,
                                                                               middle=middle)


def get_jaccard_index_zero_class_partial(middle: int):
    return partial(jaccard_index_zero_class_first_part, middle=middle), partial(jaccard_index_zero_class_second_part,
                                                                                middle=middle)


def test_showing(inp, target):
    print("Input type: ", type(inp))
    print("Target type: ", type(target))
    try:
        print("Input len: ", len(inp))
        print("Target len: ", len(target))

    except ...:
        print("Type without shape")


def f1_score():
    return partial(fbeta, thresh=0.2, beta=1)
