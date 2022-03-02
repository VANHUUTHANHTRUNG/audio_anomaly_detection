#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'cptrva'
__docformat__ = 'reStructuredText'
__all__ = [
    'extract_info_from_file_name',
    'get_files_from_dir_with_os',
    'get_files_from_dir_with_pathlib'
]

from typing import Union, List
from pathlib import Path
import os


def extract_info_from_file_name(file_name: str):
    """ Return the info of the audio sample based on its given name

    :param file_name: file name of the audio as a string
    :return: info of the audio
    """
    _, section, usage, domain, label, id_per_section = str(file_name).split('.')[0].split('_')[:6]
    return section, usage, domain, label, id_per_section


def get_files_from_dir_with_os(dir_name: str) -> List[str]:
    """Returns the files in the directory `dir_name` using the os package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[str]
    """
    return os.listdir(dir_name)


def get_files_from_dir_with_pathlib(dir_name: Union[str, Path]) -> List[Path]:
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[Path]
    """
    return sorted(list(Path(dir_name).iterdir()))


if __name__ == '__main__':

    # test_name = 'section_02_source_test_normal_0026.wav'
    test_name = 'section_00_source_train_normal_0000_strength_1_ambient.wav'
    print(extract_info_from_file_name(test_name))

    test_dir = 'data/fan'
    print(get_files_from_dir_with_pathlib(test_dir))

# EOF
