from fastai.vision import (
    nn,  # torch.nn
    create_body
)


def get_base(model):
    return nn.Sequential(
        *create_body(model)
    )


class SavedFeatures:
    feature = None

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.hook.remove()