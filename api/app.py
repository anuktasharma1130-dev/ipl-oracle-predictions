"""
api/app.py
==========
Flask REST API for IPL Oracle ML Predictions.
Exposes:
  - GET  /api/health
  - GET  /api/model-info
  - POST /api/predict

Enables Cross-Origin Resource Sharing (CORS) for local and web frontends.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from ml.prediction_engine import PredictionEngine, KNOWN_TEAMS

# Initialize Flask application
app = Flask(__name__)

# Enable CORS for all routes (allows calls from http://127.0.0.1:5000, file://, localhost, etc.)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Instantiate the singleton prediction engine
engine = PredictionEngine()


@app.route("/api/health", methods=["GET"])
def health() -> Response:
    """
    Health check endpoint.
    Returns:
        JSON object with status ok and service name.
    """
    return jsonify({
        "status": "ok",
        "service": "IPL Oracle ML API"
    }), 200


@app.route("/api/model-info", methods=["GET"])
def model_info() -> Response:
    """
    Model metadata and capability endpoint.
    Returns details on loaded models, architectures, expected inputs, and split info.
    """
    info: Dict[str, Any] = {
        "status": "ok",
        "service": "IPL Oracle ML API",
        "available_models": {
            "pre_toss": {
                "name": "pre_toss_improved_histgb",
                "model_type": "HistGradientBoostingClassifier (scikit-learn Pipeline with ColumnTransformer)",
                "expected_feature_count": len(engine.pre_toss_expected_features),
                "expected_features": engine.pre_toss_expected_features,
                "input_requirements": ["team1", "team2", "venue"],
                "optional_inputs": ["city"],
            },
            "post_toss": {
                "name": "post_toss_improved_histgb",
                "model_type": "HistGradientBoostingClassifier (scikit-learn Pipeline with ColumnTransformer)",
                "expected_feature_count": len(engine.post_toss_expected_features),
                "expected_features": engine.post_toss_expected_features,
                "input_requirements": ["team1", "team2", "venue", "toss_winner", "toss_decision"],
                "optional_inputs": ["city"],
            },
        },
        "evaluation_protocol": {
            "training_split": "2008-2023 seasons (1,029 matches)",
            "validation_split": "2024 season (71 matches)",
            "test_split": "2025-2026 seasons (134 matches, strictly held-out)",
            "leakage_safeguards": "Strict chronological state accumulation; zero future-information leakage",
        },
        "supported_teams": sorted(list(KNOWN_TEAMS)),
        "mode_selection": "Pre-toss model is selected by default; Post-toss model is automatically activated when toss_winner and toss_decision are provided.",
    }
    return jsonify(info), 200


@app.route("/api/predict", methods=["POST"])
def predict() -> Response:
    """
    Prediction endpoint.
    Accepts JSON body:
    {
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "venue": "Wankhede Stadium",
        "city": "Mumbai" (optional),
        "toss_winner": "Mumbai Indians" (optional),
        "toss_decision": "field" (optional)
    }
    """
    if not request.is_json:
        return jsonify({
            "status": "error",
            "error": "Request content-type must be application/json."
        }), 400

    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "error": "Invalid request payload. Expected a JSON object."
        }), 400

    team1 = data.get("team1")
    team2 = data.get("team2")
    venue = data.get("venue")
    city = data.get("city")
    toss_winner = data.get("toss_winner")
    toss_decision = data.get("toss_decision")

    try:
        prediction_result = engine.predict(
            team1=team1,
            team2=team2,
            venue=venue,
            city=city,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
        )
        return jsonify({
            "status": "success",
            **prediction_result
        }), 200

    except ValueError as val_err:
        return jsonify({
            "status": "error",
            "error": str(val_err)
        }), 400
    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": f"Prediction processing error: {str(exc)}"
        }), 500


if __name__ == "__main__":
    # Host on 127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=False)
