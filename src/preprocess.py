import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=13, sr=16000, frame_duration=0.5):
    y, sr = librosa.load(file_path, sr=sr)
    samples_per_frame = int(sr * frame_duration)

    features = []
    for start in range(0, len(y) - samples_per_frame + 1, samples_per_frame):
        chunk = y[start : start + samples_per_frame]
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        features.append(mfcc_mean)

    return np.array(features)