from functools import partial

from data_utils import set_data_environment

path_to_data = "/home/dmitri/Documents/Datasets/skyFinderMod"
data_dirs = set_data_environment(path_to_data, subdirs={
    "train images skyFinder": "train_skyFinder",
    "train labels skyFinder": "train_skyFinder_labels",
    "train images raw": "train_raw",
    "train labels raw": "train_raw_labels",
    "train images labeled": "train_labeled",
    "train labels labeled": "train_labeled_labels",
    "train images sky": "train_sky",
    "train labels sky": "train_sky_labels",
    "valid images skyFinder": "val_skyFinder",
    "valid labels skyFinder": "val_skyFinder_labels",
    "test images skyFinder": "test_skyFinder",
    "test labels skyFinder": "test_skyFinder_labels",
    "test iPhoneXR images": "test_iphoneXR_v",
    "test iPhoneXR labels": "test_iphoneXR_v_labels",
    "test iPhoneXR images without labels": "test_iphoneXR_h"
})
data_dirs.set_translation(
    lambda image_name: f"{image_name.stem}_L.png"
)


def get_label(image, category):
    label_subdir_name = data_dirs.get_label_subdir_from_pair(category)
    return data_dirs.get_label(image, label_subdir_name)


def get_data_paths(category):
    result = {
        "images": [],
        "labels": []
    }
    for image in data_dirs.get_subdir(category + "_images").iterdir():
        result["images"].append(image)
        result["labels"].append(get_label(image, category))
    print("Dataset size: ", len(result["images"]))
    return result


get_label_train = partial(get_label, category="train")
get_label_val = partial(get_label, category="val")


def get_label_with_context(image_file):
    return get_label(image_file, image_file.parent.stem)


__all__ = [
    data_dirs,
    get_data_paths,
    get_label_train,
    get_label_val,
    get_label_with_context
]
