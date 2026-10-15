import os
import pytest
import numpy as np
from app.app import app, load_app_config, get_model, classify_audio_stream
from src.aggregate import aggregate_chunks, aggregate_top20, aggregate_mean, aggregate_max

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_config_loading():
    cfg = load_app_config()
    assert "theta_dev" in cfg
    assert isinstance(cfg["theta_dev"], float)
    assert cfg["theta_dev"] == 0.76
    assert cfg["aggregation"] == "top20"

def test_model_loading():
    model = get_model()
    assert model is not None
    assert hasattr(model, "predict_proba")

def test_aggregate_methods():
    scores = np.array([0.1, 0.2, 0.9, 0.4, 0.8])
    assert aggregate_mean(scores) == pytest.approx(0.48)
    assert aggregate_max(scores) == pytest.approx(0.90)
    assert aggregate_top20(scores) == pytest.approx(0.90)
    assert aggregate_chunks(scores, method="top20") == pytest.approx(0.90)

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Spoofing-Aware Speaker Verification" in response.data or b"SASV" in response.data

def test_api_config(client):
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.get_json()
    assert data["theta_dev"] == 0.76
    assert data["aggregation"] == "top20"
    assert data["model_loaded"] is True

def test_analyze_no_file(client):
    response = client.post("/analyze", data={})
    assert response.status_code == 400
