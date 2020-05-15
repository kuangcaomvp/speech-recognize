import os

import numpy as np
import python_speech_features as psf
from scipy.io import wavfile

WIN_LENGTH = 0.025  # Window length in seconds.
WIN_STEP = 0.010  # The step between successive windows in seconds.
MFCC_NUM_FEATURES = 26  # Number of features to extract.
MEL_NUM_FEATURES=40


def load_sample(file_path, feature_type=None, feature_normalization=None):

    __supported_feature_types = ['mel', 'mfcc']
    __supported_feature_normalizations = ['none', 'local', 'local_scalar', 'cmvn']

    feature_type = feature_type
    feature_normalization = feature_normalization

    if feature_type not in __supported_feature_types:
        raise ValueError('Requested feature type of {} isn\'t supported.'
                         .format(feature_type))

    if feature_normalization not in __supported_feature_normalizations:
        raise ValueError('Requested feature normalization method {} is invalid.'
                         .format(feature_normalization))

    if type(file_path) is not str:
        file_path = str(file_path, 'utf-8')

    if not os.path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # Load the audio files sample rate and data.
    (sampling_rate, audio_data) = wavfile.read(file_path)

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    f_max = sampling_rate / 2.  # Maximum frequency (Nyquist rate).
    f_min = 0.  # Minimum frequency.
    n_fft = 512  # Number of samples in a frame.

    if feature_type == 'mfcc':
        sample = __mfcc(
            audio_data, sampling_rate, WIN_LENGTH, WIN_STEP, MFCC_NUM_FEATURES, n_fft, f_min, f_max
        )
    elif feature_type == 'mel':
        sample = __mel(
            audio_data, sampling_rate, WIN_LENGTH, WIN_STEP, MEL_NUM_FEATURES, n_fft, f_min, f_max
        )
    else:
        raise ValueError('Unsupported feature type')

    # Make sure that data type matches TensorFlow type.
    sample = sample.astype(np.float32)


    # Get length of the sample.
    sample_len = np.array(sample.shape[0], dtype=np.int32)

    # Apply feature normalization.
    sample = __feature_normalization(sample, feature_normalization)

    # sample = [time, NUM_FEATURES], sample_len: scalar
    return sample, sample_len


def __mfcc(audio_data, sampling_rate, win_len, win_step, num_features, n_fft, f_min, f_max):
    """Convert a wav signal into Mel Frequency Cepstral Coefficients (MFCC).
    Args:
        audio_data (np.ndarray): Wav signal.
        sampling_rate (int):  Sampling rate.
        win_len (float): Window length in seconds.
        win_step (float): Window stride in seconds.
        num_features (int): Number of features to generate.
        n_fft (int): Number of Fast Fourier Transforms.
        f_min (float): Minimum frequency to consider.
        f_max (float): Maximum frequency to consider.
    Returns:
        np.ndarray: MFCC feature vectors. Shape: [time, num_features]
    """

    # Compute MFCC features.
    mfcc = psf.mfcc(signal=audio_data, samplerate=sampling_rate, winlen=win_len, winstep=win_step,
                    numcep=num_features, nfilt=num_features, nfft=n_fft,
                    lowfreq=f_min, highfreq=f_max,
                    preemph=0.97, ceplifter=22, appendEnergy=True)

    # And the first-order differences (delta features).
    # feat_mfcc_d = psf.delta(mfcc, 2)
    # feat_mfcc_dd = psf.delta(feat_mfcc_d, 2)

    # Combine MFCC with MFCC_delta
    return mfcc


def __mel(audio_data, sampling_rate, win_len, win_step, num_features, n_fft, f_min, f_max):
    """Convert a wav signal into a logarithmically scaled mel filterbank.
    Args:
        audio_data (np.ndarray): Wav signal.
        sampling_rate (int):  Sampling rate.
        win_len (float): Window length in seconds.
        win_step (float): Window stride in seconds.
        num_features (int): Number of features to generate.
        n_fft (int): Number of Fast Fourier Transforms.
        f_min (float): Minimum frequency to consider.
        f_max (float): Maximum frequency to consider.
    Returns:
        np.ndarray: Mel-filterbank. Shape: [time, num_features]
    """
    mel = psf.logfbank(signal=audio_data, samplerate=sampling_rate, winlen=win_len,
                       winstep=win_step, nfilt=num_features, nfft=n_fft,
                       lowfreq=f_min, highfreq=f_max, preemph=0.97)
    return mel


def __feature_normalization(features, method, is_mean =True, is_vars=True):
    """Normalize the given feature vector `y`, with the stated normalization `method`.
    Args:
        features (np.ndarray):
            The signal array
        method (str):
            Normalization method:
            'local': Use local (in sample) mean and standard deviation values, and apply the
                normalization element wise, like in `global`.
            'local_scalar': Uses only the mean and standard deviation of the current sample.
                The normalization is being applied by ([sample] - mean_scalar) / std_scalar
            'none': No normalization is being applied.
    Returns:
        np.ndarray: The normalized feature vector.
    """
    if method == 'none':
        return features
    if method == 'local':
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    if method == 'local_scalar':
        # Option 'local' uses scalar values.
        return (features - np.mean(features)) / np.std(features)
    if method == 'cmvn':
        square_sums = (features ** 2).sum(axis=0)
        mean = features.mean(axis=0)
        if is_mean:
            features = np.subtract(features, mean)

        if is_vars:
            var = square_sums / features.shape[0] - mean ** 2
            std = np.maximum(np.sqrt(var), 1.0e-20)
            features = np.divide(features, std)
        return features
    raise ValueError('Invalid normalization method.')



