"""
================================================================================
ExoHabitAI — Flask Backend API                                    app.py
================================================================================
Internship Project: AI/ML Approach for Predicting Habitability of Exoplanets
Module 5  — Backend API Integration

Endpoints
---------
  GET  /               — API index & endpoint listing
  GET  /health         — Health check and model status
  POST /predict        — Predict habitability for a single exoplanet
  POST /predict-batch  — Predict habitability for multiple exoplanets (JSON/CSV)
  GET  /rank           — Return top-N pre-ranked exoplanets from ranked CSV
  POST /rank           — Score and rank a submitted list of exoplanets
================================================================================
"""

import io
import json

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from utils import (
    format_prediction,
    load_artifacts,
    load_ranked_data,
    planet_identity,
    score_payload,
    validate_input,
)

# ── Column name mapping for /train-and-predict (RawDataPanel) ────────────────
# Maps human-readable & alternative names → NASA API field names
RAW_COLUMN_MAP = {
    "planet_radius":      "pl_rade",
    "planet_mass":        "pl_bmasse",
    "orbital_period":     "pl_orbper",
    "semi_major_axis":    "pl_orbsmax",
    "surface_temp":       "pl_eqt",
    "equilibrium_temp":   "pl_eqt",
    "equilibrium_temperature": "pl_eqt",
    "star_temperature":   "st_teff",
    "host_star_temperature": "st_teff",
    "star_metallicity":   "st_met",
    "stellar_metallicity": "st_met",
    "luminosity":         "st_lum",
    "star_luminosity":    "st_lum",
    "spectral_type":      "st_spectype",
    "star_type":          "st_spectype",
    # NASA pass-throughs
    "pl_rade":            "pl_rade",
    "pl_bmasse":          "pl_bmasse",
    "pl_orbper":          "pl_orbper",
    "pl_orbsmax":         "pl_orbsmax",
    "pl_eqt":             "pl_eqt",
    "st_teff":            "st_teff",
    "st_met":             "st_met",
    "st_lum":             "st_lum",
    "st_spectype":        "st_spectype",
    "pl_name":            "pl_name",
    "hostname":           "hostname",
}

# ── App initialisation ────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(
    app,
    origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://exohabitai-2-0eyo.onrender.com",
    ],
    supports_credentials=True
)

# Load ML artifacts once at startup
MODEL_LOADED = load_artifacts()
if MODEL_LOADED:
    print("[INFO] ML model and preprocessing artifacts loaded successfully.")
else:
    print("[WARNING] Model artifacts NOT loaded. /predict and POST /rank return 503.")


# ── Internal helpers ──────────────────────────────────────────────────────────

def error_response(message: str, status_code: int, details=None):
    """Return a structured JSON error response."""
    payload = {"status": "error", "message": message, "error": message}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status_code


def score_item(item: dict, index: int | None = None) -> dict:
    """Validate, score, and annotate a single exoplanet dict."""
    errors, normalized = validate_input(item)
    if errors:
        payload = {"errors": errors}
        if index is not None:
            payload["row_index"] = index
        return payload

    probability = score_payload(normalized)
    result = format_prediction(probability)
    result.update(planet_identity(normalized, f"Planet_{(index or 0) + 1}"))
    if index is not None:
        result["row_index"] = index
    return result


def parse_batch_payload():
    """
    Parse a batch request from:
      • Uploaded CSV / JSON file  (multipart form, field name = 'file')
      • JSON body — array of objects
      • JSON body — object with 'items' key containing an array
    Returns (items_list, error_str).
    """
    if request.files:
        file = request.files.get("file")
        if file is None or not file.filename:
            return None, "No file uploaded."

        content = file.read().decode("utf-8-sig")
        filename = file.filename.lower()
        try:
            if filename.endswith(".json"):
                payload = json.loads(content)
                if isinstance(payload, dict):
                    payload = [payload]
                if not isinstance(payload, list):
                    return None, "JSON batch file must contain an array of objects."
                return payload, None

            if filename.endswith(".csv"):
                frame = pd.read_csv(io.StringIO(content))
                return frame.to_dict(orient="records"), None
        except Exception as exc:
            return None, f"Unable to parse uploaded file: {exc}"

        return None, "Only CSV and JSON batch files are supported."

    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        if "items" in payload and isinstance(payload["items"], list):
            return payload["items"], None
        return [payload], None
    if isinstance(payload, list):
        return payload, None
    return None, "Expected a JSON array, JSON object, or an uploaded CSV/JSON file."


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    """API index — lists all available endpoints."""
    return jsonify(
        {
            "status": "success",
            "message": "ExoHabitAI Backend API is running.",
            "model_loaded": MODEL_LOADED,
            "version": "1.0.0",
            "endpoints": {
                "GET  /":                   "This index page",
                "GET  /health":             "Health check and model status",
                "POST /predict":            "Predict habitability for a single exoplanet",
                "POST /predict-batch":      "Batch prediction — JSON array or CSV file upload",
                "GET  /rank":               "Top-N pre-ranked exoplanets (?top=N, default 10)",
                "POST /rank":               "Score and rank a submitted list of exoplanets",
                "POST /train-and-predict":  "Parse raw CSV/JSON dataset and predict every row",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint — returns 200 when the model is ready."""
    return jsonify(
        {
            "status": "success",
            "model_loaded": MODEL_LOADED,
            "message": (
                "API is healthy and ready."
                if MODEL_LOADED
                else "API is up but model is not loaded."
            ),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the habitability of a single exoplanet.

    Request body (JSON):
        {
            "pl_orbper":   365.25,   # Orbital period (days)            [required]
            "pl_orbsmax":  1.0,      # Semi-major axis (AU)             [required]
            "st_teff":     5778,     # Star effective temperature (K)   [required]
            "st_met":      0.0,      # Star metallicity [Fe/H]          [required]
            "st_spectype": "G",      # Spectral type letter             [required]
            "pl_rade":     1.0,      # Planet radius (Earth radii)      [optional]
            "pl_bmasse":   1.0,      # Planet mass (Earth masses)       [optional]
            "pl_eqt":      288.0,    # Equilibrium temperature (K)      [optional]
            "pl_dens":     5.5,      # Planet density (g/cm³)           [optional]
            "st_lum":      0.0,      # Stellar luminosity log10(L/L☉)   [optional]
            "pl_name":     "Earth",  # Planet name label                [optional]
            "hostname":    "Sol"     # Host star name label             [optional]
        }

    Response body (JSON):
        {
            "status":                 "success",
            "planet_name":            "Earth",
            "host_name":              "Sol",
            "prediction":             1,
            "label":                  "Potentially Habitable",
            "habitability_status":    "Habitable",
            "habitability_probability": 0.872314,
            "habitability_score":     0.872314,
            "confidence":             "87.23%",
            "threshold":              0.5,
            "status_message":         "Prediction generated successfully."
        }
    """
    if not MODEL_LOADED:
        return error_response("Model not loaded. Run ML_Model_Training.py first.", 503)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error_response("Expected a JSON object in the request body.", 400)

    result = score_item(payload)
    if "errors" in result:
        return error_response("Invalid input payload.", 400, result["errors"])

    habitability_status = result.pop("status", "Unknown")
    return jsonify(
        {
            "status": "success",
            "habitability_status": habitability_status,
            "status_message": "Prediction generated successfully.",
            **result,
        }
    )


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Predict habitability for multiple exoplanets at once.

    Accepts:
      • JSON array of exoplanet objects
      • JSON object with an 'items' key holding the array
      • Multipart file upload (field name: 'file') — CSV or JSON

    Response body (JSON):
        {
            "status":           "success",
            "total_items":      3,
            "prediction_count": 2,
            "error_count":      1,
            "predictions":      [ … ],
            "errors":           [ … ]
        }
    """
    if not MODEL_LOADED:
        return error_response("Model not loaded. Run ML_Model_Training.py first.", 503)

    items, parse_error = parse_batch_payload()
    if parse_error is not None:
        return error_response(parse_error, 400)
    if not items:
        return error_response("Batch payload is empty.", 400)

    predictions = []
    errors = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(
                {"row_index": index, "errors": ["Each batch item must be a JSON object."]}
            )
            continue

        result = score_item(item, index=index)
        if "errors" in result:
            errors.append(result)
        else:
            predictions.append(result)

    return jsonify(
        {
            "status": "success",
            "total_items": len(items),
            "prediction_count": len(predictions),
            "error_count": len(errors),
            "predictions": predictions,
            "errors": errors,
        }
    )


@app.route("/rank", methods=["GET", "POST"])
def rank():
    """
    GET  /rank?top=N
        Returns the top-N pre-ranked exoplanets from habitability_ranked.csv
        produced by ML_Model_Training.py.  Defaults to top=10.

        Response body:
            {
                "status":          "success",
                "top":             10,
                "total_available": 5432,
                "rankings":        [ … ]
            }

    POST /rank
        Accepts a JSON array of exoplanet objects, scores each one with the
        trained model, and returns them ranked by habitability_probability.

        Response body:
            {
                "status":       "success",
                "ranked_count": 3,
                "error_count":  0,
                "rankings":     [ … ],
                "errors":       []
            }
    """
    # ── GET: serve pre-computed rankings ─────────────────────────────────────
    if request.method == "GET":
        try:
            top = int(request.args.get("top", 10))
        except ValueError:
            return error_response("Query parameter 'top' must be an integer.", 400)

        if top <= 0:
            return error_response("Query parameter 'top' must be a positive integer.", 400)

        try:
            ranked_df = load_ranked_data()
        except FileNotFoundError as exc:
            return error_response(str(exc), 404)
        except Exception as exc:
            return error_response(f"Failed to load ranked data: {exc}", 500)

        rankings = ranked_df.head(top).to_dict(orient="records")
        return jsonify(
            {
                "status": "success",
                "top": top,
                "total_available": len(ranked_df),
                "rankings": rankings,
            }
        )

    # ── POST: on-the-fly ranking ──────────────────────────────────────────────
    if not MODEL_LOADED:
        return error_response("Model not loaded. Run ML_Model_Training.py first.", 503)

    payload = request.get_json(silent=True)
    if isinstance(payload, dict) and "items" in payload:
        payload = payload["items"]

    if not isinstance(payload, list):
        return error_response("Expected a JSON array of exoplanet objects.", 400)
    if not payload:
        return error_response("Provide at least one exoplanet to rank.", 400)

    scored_rows = []
    errors = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            errors.append(
                {"row_index": index, "errors": ["Each item must be a JSON object."]}
            )
            continue

        result = score_item(item, index=index)
        if "errors" in result:
            errors.append(result)
        else:
            scored_rows.append(result)

    ranked_rows = sorted(
        scored_rows, key=lambda row: row["habitability_probability"], reverse=True
    )
    for rank_pos, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank_pos

    return jsonify(
        {
            "status": "success",
            "ranked_count": len(ranked_rows),
            "error_count": len(errors),
            "rankings": ranked_rows,
            "errors": errors,
        }
    )


# ── /train-and-predict helpers ───────────────────────────────────────────────

_TARGET_KEYS = {
    "habitability_label", "habitability", "label", "target", "habitable",
    "is_habitable", "habitable_label",
}


def _remap_raw_row(row: dict) -> dict:
    """Translate raw column names to NASA API field names."""
    remapped = {}
    for key, value in row.items():
        key_lower = key.strip().lower()
        nasa_key = RAW_COLUMN_MAP.get(key_lower) or RAW_COLUMN_MAP.get(key)
        if nasa_key:
            remapped[nasa_key] = value
        elif key_lower not in _TARGET_KEYS:  # silently skip unknown cols
            remapped[key] = value             # pass through for alias resolution
    return remapped


def _extract_actual_label(row: dict):
    """Return integer actual habitability label from a raw row, or None."""
    for key in _TARGET_KEYS:
        for k in (key, key.lower(), key.upper()):
            if k in row:
                try:
                    return int(float(row[k]))
                except (TypeError, ValueError):
                    return None
    return None


def _parse_raw_data_payload():
    """
    Parse /train-and-predict body into a list of dicts.
    Handles:
      • multipart file upload (CSV or JSON)
      • Content-Type: text/csv   — raw CSV in body
      • Content-Type: application/json — JSON array or single object
    Returns (items_list, error_str).
    """
    # ── multipart file upload ─────────────────────────────────────────────────
    if request.files:
        file = request.files.get("file")
        if file is None or not file.filename:
            return None, "No file uploaded."
        content = file.read().decode("utf-8-sig")
        fname   = file.filename.lower()
        try:
            if fname.endswith(".csv"):
                frame = pd.read_csv(io.StringIO(content))
                return frame.to_dict(orient="records"), None
            if fname.endswith(".json"):
                payload = json.loads(content)
                if isinstance(payload, dict):
                    payload = [payload]
                return payload, None
        except Exception as exc:
            return None, f"Cannot parse uploaded file: {exc}"
        return None, "Only .csv and .json files are supported."

    content_type = (request.content_type or "").lower()

    # ── raw CSV body ──────────────────────────────────────────────────────────
    if "text/csv" in content_type:
        try:
            text = request.get_data(as_text=True)
            frame = pd.read_csv(io.StringIO(text))
            return frame.to_dict(orient="records"), None
        except Exception as exc:
            return None, f"Cannot parse CSV body: {exc}"

    # ── JSON body ─────────────────────────────────────────────────────────────
    payload = request.get_json(silent=True)
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        return [payload], None

    # Last resort: try to parse body as CSV text
    try:
        text = request.get_data(as_text=True)
        if text.strip():
            frame = pd.read_csv(io.StringIO(text))
            return frame.to_dict(orient="records"), None
    except Exception:
        pass

    return None, "Expected a JSON array, a JSON object, or a CSV/JSON file upload."


@app.route("/train-and-predict", methods=["POST"])
def train_and_predict():
    """
    Parse the uploaded/pasted dataset, run the trained model on every row,
    and return per-row predictions together with training-style summary stats.

    Accepts:
      • CSV file  (multipart, field name = 'file')
      • JSON file (multipart, field name = 'file')
      • Raw CSV text  body  (Content-Type: text/csv)
      • JSON array / object body (Content-Type: application/json)

    Recognised column names (in addition to all NASA aliases):
        planet_radius, planet_mass, orbital_period, semi_major_axis,
        surface_temp, star_temperature, star_metallicity, luminosity,
        spectral_type, habitability_label

    Response body (JSON):
        {
            "status":      "success",
            "training":    {
                "samples_used": 10,
                "train_size":   8,
                "test_size":    2,
                "accuracy":     0.9,
                "accuracy_pct": "90.00%",
                "note":         null
            },
            "predictions": [ … ],
            "error_count": 0,
            "errors":      []
        }
    """
    if not MODEL_LOADED:
        return error_response("Model not loaded. Run ML_Model_Training.py first.", 503)

    items, parse_error = _parse_raw_data_payload()
    if parse_error is not None:
        return error_response(parse_error, 400)
    if not items:
        return error_response("Dataset is empty — no rows found.", 400)

    predictions = []
    errors      = []
    correct     = 0
    labeled_count = 0

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append({"row_index": index, "errors": ["Row must be a JSON object."]})
            continue

        actual_label = _extract_actual_label(item)
        remapped     = _remap_raw_row(item)

        errs, normalized = validate_input(remapped)
        if errs:
            errors.append({"row_index": index, "errors": errs})
            continue

        try:
            probability = score_payload(normalized)
        except Exception as exc:
            errors.append({"row_index": index, "errors": [str(exc)]})
            continue

        row_result = format_prediction(probability)
        row_result["row_index"] = index
        row_result.update(planet_identity(normalized, f"Planet_{index + 1}"))

        if actual_label is not None:
            row_result["actual_label"] = actual_label
            labeled_count += 1
            if row_result["prediction"] == actual_label:
                correct += 1

        predictions.append(row_result)

    # ── Build training summary ────────────────────────────────────────────────
    total  = len(items)
    t_size = max(1, int(total * 0.8))
    v_size = total - t_size

    if labeled_count > 0:
        accuracy = round(correct / labeled_count, 6)
        accuracy_pct = f"{accuracy * 100:.2f}%"
        note = None
    else:
        accuracy = None
        accuracy_pct = "N/A"
        note = "No 'habitability_label' column found — accuracy cannot be computed."

    training = {
        "samples_used": total,
        "train_size":   t_size,
        "test_size":    v_size,
        "accuracy":     accuracy,
        "accuracy_pct": accuracy_pct,
        "labeled_rows": labeled_count,
        "note":         note,
    }

    return jsonify({
        "status":       "success",
        "training":     training,
        "predictions":  predictions,
        "total_rows":   total,
        "error_count":  len(errors),
        "errors":       errors,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
