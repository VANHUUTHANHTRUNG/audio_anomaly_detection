#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'cptrva'
__docformat__ = 'reStructuredText'
__all__ = [

]

import numpy as np
import librosa as lb
from typing import MutableMapping, Union, Optional
import pickle
import os
from pathlib import Path
from common import *

DEV_DATA = Path('P:\\StudentDocuments\\Documents\\audio_anomaly_detection\\dev_data')
ADD_DATA = Path('P:\\StudentDocuments\\Documents\\audio_anomaly_detection\\add_data')


def extract_mel_band_energies(audio_file: np.ndarray,
                              sr: Optional[int] = 44100,
                              n_fft: Optional[int] = 1024,
                              hop_length: Optional[int] = 512,
                              n_mels: Optional[int] = 40) \
        -> np.ndarray:
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :param sr: Sampling frequency of audio file, defaults to 44100.
    :type sr: Optional[int]
    :param n_fft: STFT window length (in samples), defaults to 1024.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 512.
    :type hop_length: Optional[int]
    :param n_mels: Number of MEL frequencies/filters to be used, defaults to 40.
    :type n_mels: Optional[int]
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    spec = lb.stft(
        y=audio_file,
        n_fft=n_fft,
        hop_length=hop_length)

    mel_filters = lb.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    return np.dot(mel_filters, np.abs(spec) ** 2)


def serialize_features_and_classes(features_and_class: MutableMapping[Union[str, Path],
                                                                      Union[np.ndarray, int]],
                                   pickle_path: str) \
        -> None:
    """Serializes the features and classes.

    :param features_and_class:
    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    :param pickle_path: Path of the pickle file
    :type pickle_path: str
    """
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(features_and_class, pickle_file)


def handle_one_file(input_file_path: Union[str, Path],
                    output_file_path: Union[str, Path]):
    features_and_classes = {}

    # gather feature and info
    audio_data, sr = lb.load(input_file_path)
    mbe = extract_mel_band_energies(audio_data, sr, n_fft=8192, hop_length=512, n_mels=256)
    file_name = str(input_file_path).split('\\')[-1]
    section, usage, domain, label, id_per_section = extract_info_from_file_name(file_name)

    # assign value for dict
    features_and_classes['features'] = mbe
    features_and_classes['section'] = section
    features_and_classes['usage'] = usage
    features_and_classes['domain'] = domain
    features_and_classes['label'] = label
    features_and_classes['id_per_section'] = id_per_section

    # save object as a pickle
    serialize_features_and_classes(features_and_classes, output_file_path)

# TODO remove constraint on only fan folder


def extract_features(data_path: Union[str, Path]):
    for machine_type_path in get_files_from_dir_with_pathlib(data_path)[:1]:
        for partition_path in get_files_from_dir_with_pathlib(machine_type_path):
            features_path = 'features_test' if 'test' in str(partition_path) else 'features_train'
            features_path = Path.joinpath(machine_type_path, features_path)
            for file_path in get_files_from_dir_with_pathlib(partition_path):
                file_name = file_path.stem
                feature_saved_path = Path.joinpath(features_path, file_name)
                handle_one_file(input_file_path=file_path,
                                output_file_path=feature_saved_path)


def main():
    extract_features(DEV_DATA)


if __name__ == '__main__':
    main()

# EOF
