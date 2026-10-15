"""
Calibration script for deepfake audio detector.
Calculates the Equal Error Rate (EER) and optimal decision threshold (theta_dev)
on the ASVspoof 2019 Dev Set scores using chunk aggregation (top20 by default).
Overwrites configs/config.json with the calibrated theta_dev and aggregation method.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.aggregate import aggregate_chunks

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_config(config):
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Updated configuration saved to: {config_path}")

def compute_eer(labels, scores):
    """
    Computes Equal Error Rate (EER) and decision threshold theta where
    False Positive Rate (FPR / false alarm) equals False Negative Rate (FNR / miss).
    
    Args:
        labels (array-like): Binary labels (0 for bonafide, 1 for spoof).
        scores (array-like): Spoof scores / probabilities (higher means spoof).
        
    Returns:
        eer (float): Equal Error Rate.
        theta (float): Operating threshold at EER.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the operating point where |FPR - FNR| is minimized
    diff = np.abs(fpr - fnr)
    best_idx = np.nanargmin(diff)

    eer = float((fpr[best_idx] + fnr[best_idx]) / 2.0)
    theta = float(thresholds[best_idx])

    return eer, theta

def calibrate(
    model_path=None,
    features_path=None,
    aggregation="top20",
    scores_dir=None
):
    config = load_config()

    # Resolve paths
    if model_path is None:
        models_dir = config.get("paths", {}).get("models_dir", "models")
        model_path = os.path.join(PROJECT_ROOT, models_dir, "rf_mfcc_s1.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(PROJECT_ROOT, "model.pkl")

    if features_path is None:
        features_dir = config.get("paths", {}).get("features_dir", "features")
        features_path = os.path.join(PROJECT_ROOT, features_dir, "mfcc.npz")

    if scores_dir is None:
        scores_dir = os.path.join(PROJECT_ROOT, config.get("paths", {}).get("scores_dir", "scores"))

    print(f"Loading trained baseline model from '{model_path}'...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    clf = joblib.load(model_path)

    print(f"Loading 2019 dataset features from '{features_path}'...")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at: {features_path}")
    data = np.load(features_path)
    X = data["X"]
    y = data["y"]
    utt_ids = data["utt_ids"]

    # Generate chunk spoof probabilities
    print(f"Predicting spoof probabilities for {len(X)} audio chunks...")
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        classes = list(clf.classes_)
        fake_idx = classes.index(1) if 1 in classes else 1
        chunk_scores = probs[:, fake_idx]
    else:
        chunk_scores = clf.predict(X).astype(float)

    # Utterance-level aggregation
    print(f"Aggregating chunk scores per utterance using method='{aggregation}'...")
    unique_utts, inverse_indices = np.unique(utt_ids, return_inverse=True)
    
    utt_score_list = []
    utt_label_list = []

    for idx, u in enumerate(unique_utts):
        mask = (inverse_indices == idx)
        u_chunk_scores = chunk_scores[mask]
        u_label = int(y[mask][0])
        
        u_agg_score = aggregate_chunks(u_chunk_scores, method=aggregation)
        utt_score_list.append(u_agg_score)
        utt_label_list.append(u_label)

    scores_array = np.array(utt_score_list, dtype=np.float64)
    labels_array = np.array(utt_label_list, dtype=np.int64)

    # Save scores
    os.makedirs(scores_dir, exist_ok=True)
    scores_df = pd.DataFrame({
        "utt_id": unique_utts,
        "score": scores_array,
        "label": labels_array
    })
    scores_csv = os.path.join(scores_dir, "asvspoof2019_dev_scores.csv")
    scores_df.to_csv(scores_csv, index=False)
    # Also save generic dev_scores.csv
    scores_df.to_csv(os.path.join(scores_dir, "dev_scores.csv"), index=False)
    print(f"Utterance scores saved to '{scores_csv}' ({len(scores_df)} utterances).")

    # Compute EER and threshold
    eer, theta_dev = compute_eer(labels_array, scores_array)

    bona_scores = scores_array[labels_array == 0]
    spoof_scores = scores_array[labels_array == 1]
    
    print("\n" + "="*50)
    print("CALIBRATION RESULTS (ASVspoof 2019 Dev Set)")
    print("="*50)
    print(f"Total Utterances:      {len(scores_array)}")
    print(f"Bonafide Samples:      {len(bona_scores)} (range: [{bona_scores.min():.4f}, {bona_scores.max():.4f}])")
    print(f"Spoof Samples:         {len(spoof_scores)} (range: [{spoof_scores.min():.4f}, {spoof_scores.max():.4f}])")
    print(f"Aggregation Strategy:  {aggregation}")
    print(f"Equal Error Rate (EER):{eer * 100:.2f}%")
    print(f"Calibrated theta_dev:  {theta_dev:.4f}")
    print("="*50)

    # Explicitly overwrite configs/config.json
    config["theta_dev"] = round(theta_dev, 4)
    config["aggregation"] = aggregation
    
    if "evaluation" not in config:
        config["evaluation"] = {}
    config["evaluation"]["theta_dev"] = round(theta_dev, 4)
    config["evaluation"]["aggregation"] = aggregation
    config["evaluation"]["metric"] = "EER"
    config["evaluation"]["eer_dev"] = round(eer, 4)

    save_config(config)
    print(f"SUCCESS: config.json updated with theta_dev={round(theta_dev, 4)} and aggregation='{aggregation}'.")
    return eer, theta_dev

if __name__ == "__main__":
    calibrate(aggregation="top20")
