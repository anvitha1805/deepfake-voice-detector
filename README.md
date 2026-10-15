# Deepfake Voice Detector (Review 3)

A deepfake audio detection system based on acoustic spectral analysis and machine learning. The pipeline extracts 13 Mel-Frequency Cepstral Coefficients (MFCCs) over 0.5-second audio frames, infers chunk-level spoof probabilities using a balanced Random Forest classifier, aggregates frame scores using a `top20` multi-chunk strategy, and applies an EER-calibrated decision threshold ($\theta_{dev} = 0.76$) to classify audio as **Real** or **Deepfake**.

---

## Architecture & Workflow

```
Audio Input (.wav, .flac)
       │
       ▼
16 kHz Resampling & 13-MFCC Extraction (0.5s chunks)
       │
       ▼
Random Forest Classifier (models/rf_mfcc_s1.pkl)
       │
       ▼
Chunk Score Aggregation (top20 average)
       │
       ▼
Threshold Decision (θ_dev = 0.76) ──► Real (< 0.76) or Deepfake (≥ 0.76)
       │
       ▼
Interactive Web Interface (templates/index.html)
```

---

## Installation & Setup

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/anvitha1805/deepfake-voice-detector.git
cd deepfake-voice-detector
```

Using Python Virtual Environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

Using Conda:
```bash
conda env create -f environment.yml
conda activate deepfake-voice-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model & Feature Pipeline (Regeneration)

The repository gitignores raw audio datasets, extracted `.npz` feature caches, and compiled `.pkl` model binaries. To generate them locally:

```bash
# 1. Generate 2021 evaluation mock structures & protocols
python src/mock_data_2021.py

# 2. Parse protocols and generate dataset splits in splits/
python src/make_splits.py

# 3. Extract 13-MFCC features into features/mfcc.npz
python src/extract_features.py

# 4. Train Random Forest model and save to models/rf_mfcc_s1.pkl
python src/train_rf.py
```

---

## Starting the Web Application

Start the Flask server using either command:

```bash
python app.py
```
*or*
```bash
python app/app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Web App Features
- **Drag & Drop Audio Upload**: Supports `.wav`, `.flac`, `.mp3`, `.m4a`, and `.ogg`.
- **Real-Time Classification**: Instantly displays **REAL** or **DEEPFAKE** prediction verdict.
- **Probability Score & Threshold Gauge**: Visual comparison of the spoof probability against operating threshold $\theta_{dev} = 0.76$.
- **Acoustic Metrics**: Inference latency (ms), frame count, and diagnostic report.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` / `POST` | Web interface / direct form upload |
| `/analyze` | `POST` | Forensic analysis endpoint (accepts multipart audio) |
| `/predict` | `POST` | Alias for `/analyze` |
| `/api/config` | `GET` | Returns active threshold ($\theta_{dev}$), aggregation strategy, and model status |

Example API Request:
```bash
curl -X POST -F "audio=@data/real/LA_T_1078395.flac" http://127.0.0.1:5000/analyze
```

---

## Running Automated Tests

Run the test suite with `pytest`:

```bash
pytest -v
```