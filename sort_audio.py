import os
import shutil

# AMRUTA: Replace the path inside the quotes below with the location of your extracted LA folder
RAW_LA_PATH = r"C:\Users\amrut\Downloads\LA\LA"

PROTOCOL_FILE = os.path.join(RAW_LA_PATH, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
AUDIO_DIR = os.path.join(RAW_LA_PATH, "ASVspoof2019_LA_train", "flac")

DEST_REAL = "data/real"
DEST_FAKE = "data/fake"

os.makedirs(DEST_REAL, exist_ok=True)
os.makedirs(DEST_FAKE, exist_ok=True)

real_count, fake_count = 0, 0
with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        file_name, key = parts[1], parts[-1]
        src_path = os.path.join(AUDIO_DIR, f"{file_name}.flac")

        if key == "bonafide" and real_count < 200:
            shutil.copy(src_path, os.path.join(DEST_REAL, f"{file_name}.flac"))
            real_count += 1
        elif key == "spoof" and fake_count < 200:
            shutil.copy(src_path, os.path.join(DEST_FAKE, f"{file_name}.flac"))
            fake_count += 1

        if real_count == 200 and fake_count == 200:
            break

print(f"Done! {real_count} real files and {fake_count} fake files are in your data folder.")