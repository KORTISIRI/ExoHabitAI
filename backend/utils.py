import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exohabit_pipeline import (  # noqa: E402
    RANKED_DATA_PATH,
    artifact_paths,
    build_inference_frame,
    build_inference_raw_frame,
    blend_probabilities,
    load_feature_metadata,
    normalize_payload,
)


_model = None
_feature_columns: List[str] | None = None
_preprocessing_profile: Dict[str, Any] | None = None


def load_artifacts() -> bool:
    global _model, _feature_columns, _preprocessing_profile

    paths = artifact_paths(PROJECT_ROOT)
    try:
        _model = joblib.load(paths["final_model"])
        _feature_columns, _preprocessing_profile = load_feature_metadata(PROJECT_ROOT)
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to load backend artifacts: {exc}")
        _model = None
        _feature_columns = None
        _preprocessing_profile = None
        return False


def get_model():
    return _model


def validate_input(payload: Dict) -> Tuple[List[str], Dict]:
    return normalize_payload(payload)


def build_feature_vector(normalized_payload: Dict) -> pd.DataFrame:
    if _feature_columns is None or _preprocessing_profile is None:
        raise RuntimeError("Model artifacts are not loaded")
    return build_inference_frame(normalized_payload, _feature_columns, _preprocessing_profile)


def build_raw_vector(normalized_payload: Dict) -> pd.DataFrame:
    if _preprocessing_profile is None:
        raise RuntimeError("Model artifacts are not loaded")
    return build_inference_raw_frame(normalized_payload, _preprocessing_profile)


def score_payload(normalized_payload: Dict) -> float:
    if _feature_columns is None or _preprocessing_profile is None or _model is None:
        raise RuntimeError("Model artifacts are not loaded")

    feature_frame = build_inference_frame(normalized_payload, _feature_columns, _preprocessing_profile)
    raw_frame = build_inference_raw_frame(normalized_payload, _preprocessing_profile)
    model_probability = float(_model.predict_proba(feature_frame)[0][1])
    blended_probability = float(blend_probabilities([model_probability], raw_frame)[0])
    return blended_probability


def planet_identity(normalized_payload: Dict, default_name: str) -> Dict[str, str]:
    return {
        "planet_name": normalized_payload.get("planet_name", default_name),
        "host_name": normalized_payload.get("host_name", "Custom Host"),
    }


def format_prediction(probability: float) -> Dict[str, Any]:
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "label": "Potentially Habitable" if prediction else "Non-Habitable",
        "status": "Habitable" if prediction else "Not Habitable",
        "habitability_probability": round(probability, 6),
        "habitability_score": round(probability, 6),
        "confidence": f"{probability * 100:.2f}%",
        "threshold": 0.5,
    }


def load_ranked_data() -> pd.DataFrame:
    if not os.path.exists(RANKED_DATA_PATH):
        raise FileNotFoundError("Ranked CSV not found. Run ML_Model_Training.py first.")
    ranked_df = pd.read_csv(RANKED_DATA_PATH)
    sort_columns = ["habitability_probability"]
    ascending = [False]
    if "rank" in ranked_df.columns:
        sort_columns.append("rank")
        ascending.append(True)
    return ranked_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
