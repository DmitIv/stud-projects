from fastai.vision import (
    models
)

from model.pytorch.utility import (
    get_base
)


def get_resnet34():
    return get_base(models.resnet34)


def get_resnet50():
    return get_base(models.resnet50)
