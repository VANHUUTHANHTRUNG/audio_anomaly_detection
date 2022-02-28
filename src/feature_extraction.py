import numpy as np
import librosa as lb
from typing import MutableMapping, Union, Optional
import pickle
import os


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


def calculate_spectrogram(dir: str, filename: str)  \
        -> np.ndarray:
    
    # Read the audio file and calculate the spectrogram
    audio_data, sr = lb.load(os.path.join(dir, filename))
    mbe = extract_mel_band_energies(audio_data, sr, n_fft=8192, hop_length=512, n_mels=256)

    return mbe


def feature_extraction(dir: str, current_anomaly: int, current_normal: int):
    # Dictionary to save the features and class
    features_and_class = {}

    # Loop through the files and get the features and class
    audio_filenames = os.listdir(dir)

    for filename in audio_filenames:
        mbe = calculate_spectrogram(dir,filename)
        if 'anomaly' in filename:
            audio_class = 1
        else:
            audio_class = 0
    
        # Save the features and classes into the dictionary.
        features_and_class['features'] = mbe
        features_and_class['class'] = audio_class

        # Create pickle file
        if audio_class == 0:
            pickle_path = os.path.join(f'{dir}_pickle','normal',f'{current_normal:04}.pickle')
            current_normal += 1
        else:
            pickle_path = os.path.join(f'{dir}_pickle','anomaly',f'{current_anomaly:04}.pickle')
            current_anomaly += 1
        
        serialize_features_and_classes(features_and_class, pickle_path)


def serialize_features_and_classes(features_and_class: MutableMapping[str, Union[np.ndarray, int]], pickle_path: str) -> None:
    """Serializes the features and classes.

    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    :param pickle_path: Path of the pickle file
    :type pickle_path: str
    """
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(features_and_class, pickle_file)


if __name__ == '__main__':
    current_normal = 0
    current_anomaly = 0
    feature_extraction('test', current_anomaly, current_normal)