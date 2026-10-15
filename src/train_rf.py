"""
Train Random Forest baseline model on 13-MFCC features for deepfake audio detection.
Uses class_weight="balanced" and random_state=1 as specified for Review 3.
Saves the compressed model to models/rf_mfcc_s1.pkl.
"""

import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def train_rf(
    features_path=None,
    model_output_path=None,
    n_estimators=100,
    class_weight="balanced",
    random_state=1,
    compress=3
):
    config = load_config()
    
    if features_path is None:
        features_dir = config.get("paths", {}).get("features_dir", "features")
        features_path = os.path.join(PROJECT_ROOT, features_dir, "mfcc.npz")
        
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file not found at: {features_path}. Please run feature extraction first.")

    if model_output_path is None:
        models_dir = config.get("paths", {}).get("models_dir", "models")
        model_output_path = os.path.join(PROJECT_ROOT, models_dir, "rf_mfcc_s1.pkl")

    print(f"Loading 13-MFCC features from '{features_path}'...")
    data = np.load(features_path)
    X = data["X"]
    y = data["y"]
    utt_ids = data.get("utt_ids", None)

    print(f"Loaded {X.shape[0]} feature chunks of dimension {X.shape[1]}.")
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Class distribution: {class_dist} (0: bonafide, 1: spoof)")

    print(f"Training Random Forest (n_estimators={n_estimators}, class_weight='{class_weight}', random_state={random_state})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X, y)

    # Evaluate on training data
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    print(f"\nTraining Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y, preds, target_names=["bonafide (0)", "spoof (1)"]))

    # Save compressed model to models/rf_mfcc_s1.pkl
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path, compress=compress)
    file_size = os.path.getsize(model_output_path)
    print(f"SUCCESS: Compressed model saved to '{model_output_path}' ({file_size:,} bytes).")

    # Also update root model.pkl for application compatibility
    root_model_path = os.path.join(PROJECT_ROOT, "model.pkl")
    joblib.dump(clf, root_model_path, compress=compress)
    print(f"Updated root deployment model: '{root_model_path}'.")

    return clf, model_output_path

if __name__ == "__main__":
    train_rf(
        n_estimators=100,
        class_weight="balanced",
        random_state=1
    )
