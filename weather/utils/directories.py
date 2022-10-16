import os

from weather.utils.config import get_main_dir


def get_dataset_dir() -> str:
    return os.path.join(get_main_dir(), "datasets")


def get_cmorph_dir() -> str:
    return guarantee_existence(os.path.join(get_dataset_dir(), "cmorph"))


def get_ecmwf_dir() -> str:
    return guarantee_existence(os.path.join(get_dataset_dir(), "ecmwf"))


def get_pickle_dir() -> str:
    return guarantee_existence(os.path.join(get_dataset_dir(), "pickle"))


def get_model_save_dir() -> str:
    return guarantee_existence(os.path.join(get_main_dir(), "model_save"))


def get_h5_dir() -> str:
    return guarantee_existence(os.path.join(get_dataset_dir(), "h5"))


def guarantee_existence(path) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(path)


if __name__ == '__main__':
    print(get_h5_dir())
