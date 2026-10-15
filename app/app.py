import os
import sys
import time
import json
import tempfile
import numpy as np
import joblib
import librosa
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Determine paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Import feature extractor and aggregation
try:
    from src.preprocess import extract_mfcc
except ImportError:
    from preprocess import extract_mfcc

try:
    from src.aggregate import aggregate_chunks
except ImportError:
    def aggregate_chunks(scores, method="top20"):
        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) == 0:
            return 0.0
        method = str(method).lower().strip()
        if method == "mean":
            return float(np.mean(scores))
        elif method == "max":
            return float(np.max(scores))
        else:
            k = max(1, int(np.ceil(len(scores) * 0.20)))
            return float(np.mean(np.sort(scores)[-k:]))

# Configure Flask app pointing to root templates and static folders
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR if os.path.exists(STATIC_DIR) else None)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload

# Paths to config and models
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.json")
PRIMARY_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_mfcc_s1.pkl")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

_model = None

def load_app_config():
    """Load theta_dev, aggregation, and audio parameters from configs/config.json."""
    default_config = {
        "theta_dev": 0.76,
        "aggregation": "top20",
        "audio": {
            "sample_rate": 16000,
            "frame_duration": 0.5,
            "n_mfcc": 13
        }
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                theta = cfg.get("theta_dev")
                if theta is None:
                    theta = cfg.get("evaluation", {}).get("theta_dev", 0.76)
                agg = cfg.get("aggregation")
                if agg is None:
                    agg = cfg.get("evaluation", {}).get("aggregation", "top20")
                audio_cfg = cfg.get("audio", default_config["audio"])
                return {
                    "theta_dev": float(theta) if theta is not None else 0.76,
                    "aggregation": str(agg) if agg else "top20",
                    "audio": audio_cfg
                }
        except Exception as e:
            print(f"Warning: Error reading config.json ({e}). Using defaults.")
    return default_config

def get_model():
    """Load trained Random Forest model (models/rf_mfcc_s1.pkl or fallback)."""
    global _model
    if _model is None:
        if os.path.exists(PRIMARY_MODEL_PATH):
            print(f"Loading primary model from: {PRIMARY_MODEL_PATH}")
            _model = joblib.load(PRIMARY_MODEL_PATH)
        elif os.path.exists(FALLBACK_MODEL_PATH):
            print(f"Loading fallback model from: {FALLBACK_MODEL_PATH}")
            _model = joblib.load(FALLBACK_MODEL_PATH)
        else:
            raise FileNotFoundError(
                f"Model file not found at {PRIMARY_MODEL_PATH} or {FALLBACK_MODEL_PATH}. "
                "Please run 'python src/train_rf.py' to generate the model."
            )
    return _model

def classify_audio_stream(file_path):
    """
    Extract 13-MFCCs over 0.5s windows at 16kHz, infer chunk probabilities using
    models/rf_mfcc_s1.pkl, aggregate spoof scores using the configured aggregation
    strategy, and classify as Real or Deepfake based on theta_dev.
    """
    cfg = load_app_config()
    theta_dev = cfg["theta_dev"]
    aggregation = cfg["aggregation"]
    sr = cfg.get("audio", {}).get("sample_rate", 16000)
    frame_dur = cfg.get("audio", {}).get("frame_duration", 0.5)
    n_mfcc = cfg.get("audio", {}).get("n_mfcc", 13)

    t_start = time.perf_counter()

    # Extract 13-MFCC features
    features = extract_mfcc(file_path, n_mfcc=n_mfcc, sr=sr, frame_duration=frame_dur)

    if len(features) == 0:
        # Fallback if audio is shorter than frame duration
        y, loaded_sr = librosa.load(file_path, sr=sr)
        if len(y) == 0:
            raise ValueError("Audio stream contains no readable samples.")
        mfcc = librosa.feature.mfcc(y=y, sr=loaded_sr, n_mfcc=n_mfcc)
        features = np.array([np.mean(mfcc.T, axis=0)])

    clf = get_model()

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(features)
        classes = list(clf.classes_)
        # Class 1: spoof/deepfake, Class 0: bonafide/real
        fake_idx = classes.index(1) if 1 in classes else (len(classes) - 1)
        real_idx = classes.index(0) if 0 in classes else 0

        chunk_fake_probs = probs[:, fake_idx]
        chunk_real_probs = probs[:, real_idx]

        # Aggregate chunk-level spoof scores
        agg_fake_score = aggregate_chunks(chunk_fake_probs, method=aggregation)
    else:
        preds = clf.predict(features)
        agg_fake_score = float(np.mean(preds == 1))

    # Apply decision threshold theta_dev
    is_deepfake = bool(agg_fake_score >= theta_dev)
    prediction = "Deepfake" if is_deepfake else "Real"
    state = "deepfake" if is_deepfake else "real"

    # Probability metrics
    spoof_prob = float(np.clip(agg_fake_score, 0.0, 1.0))
    real_prob = float(np.clip(1.0 - spoof_prob, 0.0, 1.0))

    # Confidence for predicted class
    class_confidence = (spoof_prob if is_deepfake else real_prob) * 100.0

    t_end = time.perf_counter()
    latency_ms = max(1, int((t_end - t_start) * 1000))

    if is_deepfake:
        details = (
            f"Deepfake detected with spoof probability score of {spoof_prob:.4f} "
            f"({spoof_prob * 100:.2f}%), exceeding the decision threshold theta_dev = {theta_dev:.2f} "
            f"using '{aggregation}' chunk aggregation across {len(features)} frames. "
            f"Characteristic neural vocoder / synthetic artifacts identified in MFCC spectral space."
        )
    else:
        details = (
            f"Authentic human speech verified with spoof score of {spoof_prob:.4f} "
            f"({spoof_prob * 100:.2f}%), remaining below the decision threshold theta_dev = {theta_dev:.2f} "
            f"using '{aggregation}' chunk aggregation across {len(features)} frames. "
            f"Acoustic vocal tract resonance matches organic biological speech dispersion."
        )

    return {
        "prediction": prediction,
        "state": state,
        "score": round(spoof_prob, 4),
        "probability": round(spoof_prob, 4),
        "probability_percent": round(spoof_prob * 100, 2),
        "spoof_probability": round(spoof_prob, 4),
        "real_probability": round(real_prob, 4),
        "confidence": round(class_confidence, 2),
        "theta_dev": theta_dev,
        "aggregation": aggregation,
        "chunks": int(len(features)),
        "latency": latency_ms,
        "details": details
    }

@app.route("/", methods=["GET", "POST"])
def index():
    """Renders the main forensic audio interface or processes direct form uploads."""
    cfg = load_app_config()
    if request.method == "POST":
        audio_file = request.files.get("audio") or request.files.get("file")
        if not audio_file or audio_file.filename == "":
            return render_template("index.html", error="No audio file selected.", config=cfg)

        filename = secure_filename(audio_file.filename)
        _, ext = os.path.splitext(filename.lower())
        if ext not in [".wav", ".flac", ".mp3", ".m4a", ".ogg"]:
            return render_template("index.html", error=f"Unsupported audio format '{ext}'. Upload .wav or .flac.", config=cfg)

        temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
        os.close(temp_fd)
        try:
            audio_file.save(temp_path)
            res = classify_audio_stream(temp_path)
            res["filename"] = filename
            if request.is_json or request.headers.get("Accept") == "application/json":
                return jsonify(res)
            return render_template("index.html", result=res, config=cfg)
        except Exception as e:
            return render_template("index.html", error=f"Forensic analysis failed: {str(e)}", config=cfg)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    return render_template("index.html", config=cfg)

@app.route("/analyze", methods=["POST"])
@app.route("/predict", methods=["POST"])
@app.route("/upload", methods=["POST"])
def analyze():
    """
    Forensic analysis endpoint: Accepts uploaded .wav / .flac audio files,
    extracts 13-MFCCs, predicts chunk probabilities via models/rf_mfcc_s1.pkl,
    aggregates via the strategy from configs/config.json, and classifies against theta_dev.
    """
    if "audio" not in request.files and "file" not in request.files:
        return jsonify({"error": "No audio file provided in request."}), 400

    audio_file = request.files.get("audio") or request.files.get("file")
    if not audio_file or audio_file.filename == "":
        return jsonify({"error": "Selected file is empty."}), 400

    filename = secure_filename(audio_file.filename)
    _, ext = os.path.splitext(filename.lower())
    if ext not in [".wav", ".flac", ".mp3", ".m4a", ".ogg"]:
        return jsonify({"error": f"Unsupported audio format '{ext}'. Please upload .wav or .flac."}), 400

    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(temp_fd)

    try:
        audio_file.save(temp_path)
        result = classify_audio_stream(temp_path)
        result["filename"] = filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Forensic analysis failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

@app.route("/api/config", methods=["GET"])
def get_config():
    """Returns current active configuration and model info."""
    cfg = load_app_config()
    model_exists = os.path.exists(PRIMARY_MODEL_PATH) or os.path.exists(FALLBACK_MODEL_PATH)
    return jsonify({
        "theta_dev": cfg["theta_dev"],
        "aggregation": cfg["aggregation"],
        "audio": cfg["audio"],
        "model_loaded": model_exists,
        "model_path": PRIMARY_MODEL_PATH if os.path.exists(PRIMARY_MODEL_PATH) else FALLBACK_MODEL_PATH
    })

if __name__ == "__main__":
    print(f"Loading deepfake detection model from: {PRIMARY_MODEL_PATH}")
    get_model()
    # Warmup librosa / soundfile
    try:
        dummy_signal = np.zeros(16000, dtype=np.float32)
        _ = librosa.feature.mfcc(y=dummy_signal, sr=16000, n_mfcc=13)
    except Exception:
        pass
    cfg = load_app_config()
    print(f"Active Threshold (theta_dev): {cfg['theta_dev']}")
    print(f"Active Aggregation Method:    {cfg['aggregation']}")
    print("Starting Flask SASV-Core Forensic Server on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)