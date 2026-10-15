"""
Generate mock/dummy dataset structure and protocol files for ASVspoof 2021 LA and 2021 DF.
Creates dummy directory structures, protocol txt files, and exactly 10 silent/dummy .flac files
(5 for LA, 5 for DF) inside the data/ directory, without altering existing 2019 data.
"""

import os
import soundfile as sf
import numpy as np

def generate_mock_data(base_data_dir="data", n_flacs_per_set=5, sample_rate=16000, duration=4.0):
    print(f"Generating mock 2021 datasets in '{base_data_dir}'...")

    # 1. Setup directory structures
    la_dir = os.path.join(base_data_dir, "ASVspoof2021_LA_eval")
    la_flac_dir = os.path.join(la_dir, "flac")
    df_dir = os.path.join(base_data_dir, "ASVspoof2021_DF_eval")
    df_flac_dir = os.path.join(df_dir, "flac")
    protocols_dir = os.path.join(base_data_dir, "protocols")

    os.makedirs(la_flac_dir, exist_ok=True)
    os.makedirs(df_flac_dir, exist_ok=True)
    os.makedirs(protocols_dir, exist_ok=True)

    # 2. Generate exactly 10 silent/dummy .flac files (5 for LA, 5 for DF)
    # Audio with very subtle signal (low amplitude 440Hz sine wave + dither) so DSP transforms don't fail on log(0)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    dummy_audio = 1e-4 * np.sin(2 * np.pi * 440 * t) + 1e-5 * np.random.normal(0, 1, len(t)).astype(np.float32)

    la_flac_files = []
    for i in range(n_flacs_per_set):
        flac_name = f"LA_E_{i:07d}.flac"
        flac_path = os.path.join(la_flac_dir, flac_name)
        sf.write(flac_path, dummy_audio, sample_rate, format="FLAC")
        la_flac_files.append(flac_name)

    df_flac_files = []
    for i in range(n_flacs_per_set):
        flac_name = f"DF_E_{i:07d}.flac"
        flac_path = os.path.join(df_flac_dir, flac_name)
        sf.write(flac_path, dummy_audio, sample_rate, format="FLAC")
        df_flac_files.append(flac_name)

    total_flacs = len(la_flac_files) + len(df_flac_files)
    print(f"Generated {len(la_flac_files)} dummy FLACs for 2021 LA and {len(df_flac_files)} for 2021 DF (total: {total_flacs}).")

    # 3. Generate ASVspoof 2021 LA protocol txt file
    # Standard format: [speaker_id] [file_name] [codec] [source] [attack_id] [key] [trim]
    # We generate 5000 real and 5000 fake lines referencing the dummy audio files
    la_protocol_path = os.path.join(la_dir, "ASVspoof2021.LA.cm.eval.trl.txt")
    la_proto_copy = os.path.join(protocols_dir, "ASVspoof2021.LA.cm.eval.trl.txt")
    with open(la_protocol_path, "w", encoding="utf-8") as f:
        for i in range(5000):
            utt_id = f"LA_E_{i:07d}"
            # Real / bonafide line
            f.write(f"LA_{1000 + (i % 20)} {utt_id} alaw uncompressed - bonafide notrim\n")
        for i in range(5000, 10000):
            utt_id = f"LA_E_{i:07d}"
            attack = f"A{(i % 13) + 7:02d}" # A07 - A19
            # Fake / spoof line
            f.write(f"LA_{1000 + (i % 20)} {utt_id} alaw uncompressed {attack} spoof notrim\n")

    # Mirror copy in protocols dir
    with open(la_proto_copy, "w", encoding="utf-8") as f:
        with open(la_protocol_path, "r", encoding="utf-8") as src_f:
            f.write(src_f.read())

    # 4. Generate ASVspoof 2021 DF protocol txt file
    # Standard format: [speaker_id] [file_name] [codec] [source] [attack_id] [key] [trim]
    df_protocol_path = os.path.join(df_dir, "ASVspoof2021.DF.cm.eval.trl.txt")
    df_proto_copy = os.path.join(protocols_dir, "ASVspoof2021.DF.cm.eval.trl.txt")
    with open(df_protocol_path, "w", encoding="utf-8") as f:
        for i in range(5000):
            utt_id = f"DF_E_{i:07d}"
            # Real / bonafide line
            f.write(f"DF_{2000 + (i % 20)} {utt_id} mp3 high - bonafide notrim\n")
        for i in range(5000, 10000):
            utt_id = f"DF_E_{i:07d}"
            attack = f"A{(i % 13) + 7:02d}"
            # Fake / spoof line
            f.write(f"DF_{2000 + (i % 20)} {utt_id} mp3 high {attack} spoof notrim\n")

    # Mirror copy in protocols dir
    with open(df_proto_copy, "w", encoding="utf-8") as f:
        with open(df_protocol_path, "r", encoding="utf-8") as src_f:
            f.write(src_f.read())

    print(f"Protocol files successfully created:")
    print(f"  - {la_protocol_path} (10,000 entries: 5,000 real, 5,000 fake)")
    print(f"  - {df_protocol_path} (10,000 entries: 5,000 real, 5,000 fake)")
    print("Mock data generation complete!")

if __name__ == "__main__":
    generate_mock_data()
