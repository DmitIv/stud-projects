from pathlib import (
    Path
)


class DataEnv:
    def __init__(self):
        self._data_dirs: dict = {}
        self._image2label_names_mapping: callable = None
        self._reverse_index: dict = {}

    def set_root(self, root_dir: str) -> None:
        self._data_dirs["root"] = Path(root_dir)
        self._reverse_index[Path(root_dir)] = "root"

    def set_subdir(self, subdir: str, subdir_name: str = None) -> None:
        name = subdir
        if subdir_name is not None:
            name = subdir_name
        self._data_dirs[name] = self._data_dirs["root"] / subdir
        self._reverse_index[subdir] = name

    def set_translation(self, map_func: callable):
        self._image2label_names_mapping = map_func

    def get_root(self) -> Path:
        return self._data_dirs["root"]

    def get_subdir(self, subdir_name: str = None) -> Path:
        return self._data_dirs[subdir_name]

    def get_label_name(self, image: Path) -> str:
        return self._image2label_names_mapping(image)

    def get_label(self, image: Path, labels_subdir_name: Path):
        return self._data_dirs[labels_subdir_name] / self.get_label_name(image)

    def get_label_subdir_from_pair(self, image_subdir: str):
        if image_subdir in self._reverse_index.keys():
            return self._reverse_index[image_subdir].replace("images", "labels")
        raise RuntimeError("Label subdir for this image dir does not exist")


def set_data_environment(data_dir: str, subdirs: dict) -> DataEnv:
    data_env = DataEnv()
    data_env.set_root(data_dir)
    for key, value in subdirs.items():
        data_env.set_subdir(value, key)

    return data_env
