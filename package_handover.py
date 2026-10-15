import os
import zipfile

FILES_TO_PACK = [
    "src/preprocess.py",
    "src/train.py",
    "model.pkl",
    "requirements.txt",
    "environment.yml",
    "README.md",
    "Deepfake_Audio_Detector.pptx",
]

OUTPUT_ZIP = "handover_files.zip"

def create_archive():
    print(f"Creating archive '{OUTPUT_ZIP}'...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in FILES_TO_PACK:
            if os.path.exists(file_path):
                # Standardize arcname with forward slashes for clean structure inside zip
                arcname = file_path.replace("\\", "/")
                zip_file.write(file_path, arcname=arcname)
                size = os.path.getsize(file_path)
                print(f"  [+] Added: {arcname} ({size:,} bytes)")
            else:
                print(f"  [!] Warning: File not found: {file_path}")

    print(f"\nSuccessfully created '{OUTPUT_ZIP}' ({os.path.getsize(OUTPUT_ZIP):,} bytes).")

if __name__ == "__main__":
    create_archive()
