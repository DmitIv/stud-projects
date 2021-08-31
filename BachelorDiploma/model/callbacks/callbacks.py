from functools import partial
from fastai.callbacks.tensorboard import (
    LearnerTensorboardWriter
)
from datetime import datetime
from pathlib import Path


def tensorboard_cb(log_dir: str, log_name: str = None):
    now = datetime.now()
    name = now.strftime("%d_%m_%Y_%H_%M_%S")
    if log_name is not None:
        name = log_name + "_" + name
    return partial(
        LearnerTensorboardWriter,
        base_dir=Path(log_dir),
        name=name
    )
