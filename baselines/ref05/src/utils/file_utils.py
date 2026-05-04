import json
import logging
import os
from shutil import SameFileError
from typing import BinaryIO, List, Optional

from fsspec.core import url_to_fs
from lightning.fabric.utilities.types import _PATH
from pyarrow import fs as pyarrow_fs

from src.utils.decorators import retry


@retry()
def get_file_size(file_path: str) -> int:
    fs, _ = url_to_fs(file_path)
    return fs.size(file_path)


@retry()
def copy_to_remote(local_path: str, remote_path: str, recursive: bool = True) -> None:
    try:
        logging.info(f"Copying {local_path} to {remote_path}")
        fs, _ = url_to_fs(remote_path)
        fs.put(local_path, remote_path, recursive=recursive)
        logging.info(f"Finished copying {local_path} to {remote_path}")
    except SameFileError:
        logging.warning(f"{local_path} and {remote_path} are the same. Skipping copy.")


@retry()
def file_exists_local_or_remote(file_path: str) -> bool:
    fs, _ = url_to_fs(file_path)
    return fs.exists(file_path)


@retry()
def open_local_or_remote(file_path: str, mode: str = "r") -> BinaryIO:
    fs, _ = url_to_fs(file_path)
    return fs.open(file_path, mode)


def load_json(file_path: str) -> dict:
    with open_local_or_remote(file_path, "r") as f:
        feature_map = json.load(f)
    return feature_map


@retry()
def open_pyarrow_file(file_path: str):


    fs, path = pyarrow_fs.FileSystem.from_uri(file_path)
    return fs.open_input_file(path)


def get_last_modified_file(
    folder_path: str, suffix="*", should_update_prefix=True
) -> str:


    fs, _ = url_to_fs(folder_path)
    file_list = list_files(folder_path, suffix, should_update_prefix)
    if not file_list:
        return ""

    latest_mtime = 0
    for file in file_list:
        info = fs.info(file)
        mtime = info.get("mtime", 0)
        if latest_mtime == 0 or mtime > latest_mtime:
            latest_mtime = mtime
            latest_file = file
    return latest_file


def remove_file_extension(path: _PATH) -> _PATH:


    base, _ = os.path.splitext(path)
    return base


def has_no_extension(filepath: _PATH) -> bool:

    filename = os.path.basename(filepath)

    _, extension = os.path.splitext(filename)
    return extension == ""


def list_subfolders(
    directory_path,
    should_update_prefix: bool = True,
):


    fs, _ = url_to_fs(directory_path)


    all_items = fs.ls(directory_path)


    folders = [
        f"{fs.protocol[0]}://{item}" if should_update_prefix else item
        for item in all_items
        if fs.isdir(item) and item != directory_path
    ]

    return folders


@retry()
def list_files(
    folder_path: str,
    suffix: str = "*",


    should_update_prefix: bool = True,
) -> List[str]:


    folder_path = folder_path.removesuffix("/")

    fs, _ = url_to_fs(folder_path)
    return (

        [f"{fs.protocol[0]}://{x}" for x in fs.glob(f"{folder_path}/{suffix}")]
        if should_update_prefix
        else fs.glob(f"{folder_path}/{suffix}")
    )


def replace_char_after_segment(
    path: str,
    char_to_replace: str,
    replacement_char: str,
    segment_to_find: Optional[str] = None,
) -> str:


    if segment_to_find is None:
        return path.replace(char_to_replace, replacement_char)


    segment_index = path.find(segment_to_find)

    if segment_index != -1:

        segment_end = segment_index + len(segment_to_find)
        before_segment = path[:segment_end]
        after_segment = path[segment_end:]


        modified_after_segment = after_segment.replace(
            char_to_replace, replacement_char
        )


        return before_segment + modified_after_segment


    return path
