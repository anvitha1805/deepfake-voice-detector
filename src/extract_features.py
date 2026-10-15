"""
Extract 13-MFCCs or CQCCs over 0.5s windows at 16kHz for audio files in splits/ or data/.
Supports --trim ends flag using librosa.effects.trim (top_db=30) and --feature cqcc flag.
Saves extracted feature arrays as .npz files in the features/ directory.
"""

import os
import sys
import argparse
import numpy as np
import librosa
import scipy.fftpack
import pandas as pd

# Add project root and src to sys.path to import preprocess
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from preprocess import extract_mfcc
except ImportError:
    from src.preprocess import extract_mfcc

def extract_cqcc_for_audio(y, sr=16000, n_cqcc=13, frame_duration=0.5):
    """Extract CQCCs over frame_duration windows at sr Hz."""
    samples_per_frame = int(sr * frame_duration)
    features = []

    for start in range(0, len(y) - samples_per_frame + 1, samples_per_frame):
        chunk = y[start : start + samples_per_frame]
        cqt = np.abs(librosa.cqt(chunk, sr=sr, n_bins=60, bins_per_octave=12, fmin=130))
        log_cqt = np.log(cqt + 1e-8)
        cqcc = scipy.fftpack.dct(log_cqt, axis=0, norm="ortho")[:n_cqcc]
        cqcc_mean = np.mean(cqcc.T, axis=0)
        features.append(cqcc_mean)

    if len(features) == 0 and len(y) > 0:
        cqt = np.abs(librosa.cqt(y, sr=sr, n_bins=60, bins_per_octave=12, fmin=130))
        log_cqt = np.log(cqt + 1e-8)
        cqcc = scipy.fftpack.dct(log_cqt, axis=0, norm="ortho")[:n_cqcc]
        features.append(np.mean(cqcc.T, axis=0))

    return np.array(features)

def extract_features_single_file(file_path, feature_type="mfcc", trim="none", sr=16000, n_coeffs=13, frame_duration=0.5, top_db=30):
    """Extract features (MFCC or CQCC) for a single audio file with optional trimming."""
    if not os.path.exists(file_path):
        # Check relative to PROJECT_ROOT
        alt_path = os.path.join(PROJECT_ROOT, file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Audio file not found: {file_path}")

    # If MFCC with no trimming, use base logic directly from preprocess.py
    if feature_type == "mfcc" and trim == "none":
        return extract_mfcc(file_path, n_mfcc=n_coeffs, sr=sr, frame_duration=frame_duration)

    # Otherwise load audio
    y, sr = librosa.load(file_path, sr=sr)

    # Optional trim ends using librosa.effects.trim(top_db=30)
    if trim == "ends":
        y, _ = librosa.effects.trim(y, top_db=top_db)

    samples_per_frame = int(sr * frame_duration)

    if feature_type == "mfcc":
        features = []
        for start in range(0, len(y) - samples_per_frame + 1, samples_per_frame):
            chunk = y[start : start + samples_per_frame]
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_coeffs)
            features.append(np.mean(mfcc.T, axis=0))
        if len(features) == 0 and len(y) > 0:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_coeffs)
            features.append(np.mean(mfcc.T, axis=0))
        return np.array(features)

    elif feature_type == "cqcc":
        return extract_cqcc_for_audio(y, sr=sr, n_cqcc=n_coeffs, frame_duration=frame_duration)

    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

def load_data_manifest(csv_path="splits/asvspoof2019_train.csv"):
    """Load audio manifest from CSV or fallback to data directory."""
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, csv_path)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded manifest from '{csv_path}' with {len(df)} records.")
        return df

    # Fallback to scanning data/real and data/fake
    rows = []
    for lbl, folder in [("bonafide", "real"), ("spoof", "fake")]:
        dir_path = os.path.join(PROJECT_ROOT, "data", folder)
        if os.path.exists(dir_path):
            for f in sorted(os.listdir(dir_path)):
                if f.lower().endswith((".flac", ".wav")):
                    rows.append({
                        "utt_id": os.path.splitext(f)[0],
                        "path": f"data/{folder}/{f}",
                        "label": lbl
                    })
    return pd.DataFrame(rows)

def run_extraction(feature="mfcc", trim="none", split_csv="splits/asvspoof2019_train.csv", output_path=None, features_dir="features"):
    features_dir = os.path.join(PROJECT_ROOT, features_dir)
    os.makedirs(features_dir, exist_ok=True)

    df = load_data_manifest(split_csv)
    if df.empty:
        print("No audio samples found to extract features from.")
        return

    X_list, y_list, utt_list = [], [], []
    cache = {}

    print(f"Extracting '{feature.upper()}' features (trim='{trim}') for {len(df)} audio records...")

    for idx, row in df.iterrows():
        path = str(row["path"])
        label_str = str(row.get("label", "bonafide")).lower()
        utt_id = str(row.get("utt_id", os.path.basename(path)))

        # 0 for bonafide/real, 1 for spoof/fake
        int_label = 1 if label_str in ["spoof", "fake", "1"] else 0

        # Cache feature extraction by audio path to prevent redundant loads
        if path not in cache:
            try:
                feats = extract_features_single_file(path, feature_type=feature, trim=trim)
                cache[path] = feats
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}")
                cache[path] = np.empty((0, 13))

        feats = cache[path]
        for f in feats:
            X_list.append(f)
            y_list.append(int_label)
            utt_list.append(utt_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    utt_ids = np.array(utt_list, dtype=str)

    print(f"Extracted {len(X)} feature frames of dimension {X.shape[1] if len(X) > 0 else 0}.")

    # Determine output file path
    if output_path is None:
        trim_suffix = "_trimmed" if trim == "ends" else ""
        out_filename = f"{feature}{trim_suffix}.npz"
        output_path = os.path.join(features_dir, out_filename)
    elif not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    # Save outputs as .npz arrays in features directory
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        features=X,
        labels=y,
        utt_ids=utt_ids
    )
    print(f"SUCCESS: Features saved to '{output_path}'.")

    # Also save standard aliases for maximum compatibility
    if trim == "ends":
        alt_names = [f"{feature}_ends.npz", f"train_{feature}_trimmed.npz"]
    else:
        alt_names = [f"train_{feature}.npz", f"features_{feature}.npz"]

    for alt in alt_names:
        alt_path = os.path.join(features_dir, alt)
        if not os.path.exists(alt_path):
            try:
                np.savez_compressed(alt_path, X=X, y=y, features=X, labels=y, utt_ids=utt_ids)
            except Exception:
                pass

    return output_path

def main():
    parser = argparse.ArgumentParser(description="Extract MFCC or CQCC features from audio files.")
    parser.add_argument("--feature", choices=["mfcc", "cqcc"], default="mfcc", help="Feature extraction type (default: mfcc)")
    parser.add_argument("--trim", choices=["none", "ends"], default="none", help="Trim silence from audio ends (top_db=30)")
    parser.add_argument("--split", default="splits/asvspoof2019_train.csv", help="Split CSV to process")
    parser.add_argument("--output", default=None, help="Output .npz path")
    parser.add_argument("--features_dir", default="features", help="Output directory for features")

    args = parser.parse_args()
    run_extraction(
        feature=args.feature,
        trim=args.trim,
        split_csv=args.split,
        output_path=args.output,
        features_dir=args.features_dir
    )

if __name__ == "__main__":
    main()
