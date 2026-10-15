"""
Root entrypoint for Deepfake Voice Detector Flask Application.
Permits starting server directly with 'python app.py' or 'python app/app.py'.
"""
import os
import sys
import importlib.util

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_MODULE_PATH = os.path.join(PROJECT_ROOT, "app", "app.py")

spec = importlib.util.spec_from_file_location("deepfake_app_server", APP_MODULE_PATH)
_flask_module = importlib.util.module_from_spec(spec)
sys.modules["deepfake_app_server"] = _flask_module
spec.loader.exec_module(_flask_module)

app = _flask_module.app
classify_audio_stream = _flask_module.classify_audio_stream
load_app_config = _flask_module.load_app_config
get_model = _flask_module.get_model

if __name__ == "__main__":
    get_model()
    cfg = load_app_config()
    print(f"Active Threshold (theta_dev): {cfg['theta_dev']}")
    print(f"Active Aggregation Method:    {cfg['aggregation']}")
    print("Starting Flask SASV-Core Forensic Server on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
