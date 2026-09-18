import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from preprocess import extract_mfcc

def load_dataset():
    X, y = [], []

    real_dir = "data/real"
    for fname in os.listdir(real_dir):
        if fname.endswith((".flac", ".wav")):
            feats = extract_mfcc(os.path.join(real_dir, fname))
            for f in feats:
                X.append(f)
                y.append(0)

    fake_dir = "data/fake"
    for fname in os.listdir(fake_dir):
        if fname.endswith((".flac", ".wav")):
            feats = extract_mfcc(os.path.join(fake_dir, fname))
            for f in feats:
                X.append(f)
                y.append(1)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Extracting features from audio files...")
    X, y = load_dataset()

    print(f"Training Random Forest on {len(X)} audio chunks...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Training Accuracy: {acc * 100:.2f}%")

    joblib.dump(clf, "model.pkl")
    print("SUCCESS: Model saved as model.pkl")
    