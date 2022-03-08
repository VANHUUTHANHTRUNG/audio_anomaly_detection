#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'cptrva'
__docformat__ = 'reStructuredText'
__all__ = [
    'handle_one_file',
    'extract_features',
    'dataframe_serialize',
    'serialize_features_and_classes',
    'extract_mel_band_energies',

]

import numpy as np
import librosa as lb
import pandas as pd
from typing import MutableMapping, Union, Optional
import pickle
import os
from pathlib import Path
import pandas as pd

from common import *
DEV_DATA = Path('dev_data') # This line is for running on MacOS
# DEV_DATA = Path('P:\\StudentDocuments\\Documents\\audio_anomaly_detection\\dev_data')
# ADD_DATA = Path('P:\\StudentDocuments\\Documents\\audio_anomaly_detection\\add_data')


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
                    output_file_path: Union[str, Path]) -> None:
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
    features_and_classes['label'] = 0 if label == 'normal' else 1
    features_and_classes['id_per_section'] = id_per_section

    # save object as a pickle
    serialize_features_and_classes(features_and_classes, output_file_path)


def extract_features(data_path: Union[str, Path]) -> None:
    for machine_type_path in get_files_from_dir_with_pathlib(data_path):
        print("Machine in", machine_type_path)
        # Ignore the feature_x folder in case we want to overwrite pickle files
        for partition_path in filter(lambda p: ('features' not in str(p)) and ('df' not in str(p)),
                                     get_files_from_dir_with_pathlib(machine_type_path)):
            features_path = 'features_test' if 'test' in str(partition_path) else 'features_train'
            features_path = Path.joinpath(machine_type_path, features_path)
            print("Save features in ", features_path)
            for file_path in get_files_from_dir_with_pathlib(partition_path):
                file_name = file_path.stem
                feature_saved_path = Path.joinpath(features_path, file_name)
                handle_one_file(input_file_path=file_path,
                                output_file_path=feature_saved_path)
        print("--------------------------")
        return


def dataframe_serialize(data_path: Union[str, Path]) -> None:
    """ Group the features and info of all sample into one pandas data frame

    :param data_path: path to data folder (dev_data or add_data)
    """
    for machine_type_path in get_files_from_dir_with_pathlib(data_path):
        print("Machine in", machine_type_path)
        for partition_path in filter(lambda p: 'features' in str(p),
                                     get_files_from_dir_with_pathlib(machine_type_path)):
            df_path = 'df_test' if 'test' in str(partition_path) else 'df_train'
            df_path = Path.joinpath(machine_type_path, df_path)
            print("Save df in ", df_path)
            dataframe = pd.DataFrame([], columns=['section', 'label', 'features'])
            for pickle_path in get_files_from_dir_with_pathlib(partition_path):
                file_name = pickle_path.stem
                print(pickle_path)
                pickle_read = pd.read_pickle(pickle_path)
                df = pd.DataFrame([{'section': pickle_read['section'],
                                    'label': pickle_read['label'],
                                    'features': pickle_read['features']}])
                dataframe = pd.concat(dataframe, df)
            dataframe['id'] = dataframe.index
            df_filename = 'df_test.pkl' if 'test' in str(df_path) else 'df_train.pkl'
            dataframe.to_pickle(Path.joinpath(df_path, df_filename))

        print('Done with ', machine_type_path.name)
        print('------------------------------')
        return


def get_info_one_file(input_file_path: Union[str, Path]):
    features_and_classes = {}

    # gather feature and info
    audio_data, sr = lb.load(input_file_path)
    mbe = extract_mel_band_energies(audio_data, sr, n_fft=8192, hop_length=512, n_mels=256)
    # file_name = str(input_file_path).split('\\')[-1]
    file_name = str(input_file_path).split('/')[-1]
    section, _, _, label, _ = extract_info_from_file_name(file_name)

    # assign value for dict
    features_and_classes['Feature'] = mbe
    features_and_classes['Section'] = section
    features_and_classes['Label'] = label

    # Return information dict
    return features_and_classes


def extract_features_dataframe(data_path: Union[str, Path]):
    data_df = pd.DataFrame(columns=['Section', 'Label', 'Feature'])

    for machine_type_path in get_files_from_dir_with_pathlib(data_path)[:1]:
        for partition_path in get_files_from_dir_with_pathlib(machine_type_path):
            machine_type = str(partition_path).split('/')[-2]
            partition = str(partition_path).split('/')[-1]
            # split = 'test' if 'test' in str(partition_path) else 'train'
            # Add data to dataframe
            for file_path in get_files_from_dir_with_pathlib(partition_path):
                features_and_classes = get_info_one_file(file_path)
                data_df = data_df.append(features_and_classes, ignore_index=True)
    
            # Save dataframe to a pickle file
            pickle_path = f"dev_data_df/{machine_type}/{partition}.pickle"
            data_df.to_pickle(pickle_path)


def main():
    extract_features(ADD_DATA)
    dataframe_serialize(ADD_DATA)
    # extract_features(DEV_DATA)
    extract_features_dataframe(DEV_DATA)


if __name__ == '__main__':
    main()

# EOF
