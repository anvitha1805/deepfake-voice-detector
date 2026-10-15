"""
Generate 64-bin, 4-second log-mel spectrograms for audio anti-spoofing detection.
Uses librosa to compute mel spectrograms and converts them to decibel log scale.
Supports pad/crop to fixed 4.0s (64,000 samples at 16kHz) and optional trimming.
"""

import os
import sys
import argparse
import numpy as np
import librosa
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def pad_or_crop_audio(y, target_samples=64000):
    """Pads or crops 1D audio array to exact target_samples length."""
    length = len(y)
    if length < target_samples:
        pad_width = target_samples - length
        y = np.pad(y, (0, pad_width), mode="constant", constant_values=0)
    elif length > target_samples:
        y = y[:target_samples]
    return y

def generate_logmel(
    audio_input,
    sr=16000,
    duration=4.0,
    n_mels=64,
    n_fft=512,
    hop_length=160,
    trim=False,
    top_db=30
):
    """
    Generate a 64-bin, 4.0s log-mel spectrogram.

    Parameters:
        audio_input: str (file path) or np.ndarray (audio waveform)
        sr: int, sampling rate in Hz (default: 16000)
        duration: float, audio length in seconds (default: 4.0s)
        n_mels: int, number of Mel frequency bands (default: 64)
        n_fft: int, FFT window size (default: 512)
        hop_length: int, hop length in samples (default: 160)
        trim: bool, whether to trim silence from ends (default: False)
        top_db: float, threshold for trimming in dB (default: 30)

    Returns:
        np.ndarray: log-mel spectrogram of shape (n_mels, time_steps)
    """
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            alt_path = os.path.join(PROJECT_ROOT, audio_input)
            if os.path.exists(alt_path):
                audio_input = alt_path
            else:
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
        y, sr = librosa.load(audio_input, sr=sr)
    else:
        y = np.asarray(audio_input, dtype=np.float32)

    # Optional trim
    if trim:
        y, _ = librosa.effects.trim(y, top_db=top_db)

    # Pad or crop to exact 4.0s (64,000 samples at 16kHz)
    target_samples = int(sr * duration)
    y = pad_or_crop_audio(y, target_samples=target_samples)

    # Compute Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=sr // 2,
        power=2.0
    )

    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def batch_generate_logmels(
    split_csv="splits/asvspoof2019_train.csv",
    output_path="features/logmel.npz",
    duration=4.0,
    n_mels=64,
    n_fft=512,
    hop_length=160,
    sr=16000,
    trim=False
):
    """Batch process audio files from a split CSV and save as .npz."""
    if not os.path.isabs(split_csv):
        split_csv = os.path.join(PROJECT_ROOT, split_csv)

    if not os.path.exists(split_csv):
        print(f"Error: Split file '{split_csv}' not found.")
        return None

    df = pd.read_csv(split_csv)
    print(f"Processing {len(df)} samples from '{split_csv}' for 64-bin, 4s log-mel generation...")

    specs, labels, utt_ids = [], [], []
    cache = {}

    for _, row in df.iterrows():
        path = str(row["path"])
        label_str = str(row.get("label", "bonafide")).lower()
        utt_id = str(row.get("utt_id", os.path.basename(path)))
        int_label = 1 if label_str in ["spoof", "fake", "1"] else 0

        if path not in cache:
            try:
                spec = generate_logmel(
                    path,
                    sr=sr,
                    duration=duration,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    trim=trim
                )
                cache[path] = spec
            except Exception as e:
                print(f"Warning: Failed to generate logmel for {path}: {e}")
                cache[path] = np.zeros((n_mels, int(sr * duration // hop_length) + 1), dtype=np.float32)

        specs.append(cache[path])
        labels.append(int_label)
        utt_ids.append(utt_id)

    X = np.array(specs, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    utts = np.array(utt_ids, dtype=str)

    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, features=X, labels=y, utt_ids=utts)
    print(f"SUCCESS: Saved {len(X)} log-mel spectrograms of shape {X.shape[1:]} to '{output_path}'.")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate 64-bin, 4s log-mel spectrograms.")
    parser.add_argument("--split", default="splits/asvspoof2019_train.csv", help="Split CSV path")
    parser.add_argument("--output", default="features/logmel.npz", help="Output .npz path")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration in seconds (default: 4.0)")
    parser.add_argument("--n_mels", type=int, default=64, help="Number of Mel frequency bins (default: 64)")
    parser.add_argument("--trim", action="store_true", help="Trim silence from audio ends")

    args = parser.parse_args()
    batch_generate_logmels(
        split_csv=args.split,
        output_path=args.output,
        duration=args.duration,
        n_mels=args.n_mels,
        trim=args.trim
    )

if __name__ == "__main__":
    main()
