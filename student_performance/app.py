"""Flask JSON API for student performance predictions."""

import math
from pathlib import Path

from flask import Flask, Response, jsonify, request

from .model import load_model, predict_outcome
from .paths import DEFAULT_MODEL_PATH

REQUIRED_FIELDS = {"study_hours", "attendance_percentage"}
MAX_REQUEST_BODY_BYTES = 4096


def create_app(model_path: str | Path = DEFAULT_MODEL_PATH) -> Flask:
    """Create the API and load its persisted model once at startup."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BODY_BYTES
    model = load_model(model_path)

    @app.errorhandler(413)
    def request_too_large(_error: Exception) -> tuple[Response, int]:
        return jsonify(error="Request body is too large."), 413

    @app.post("/predict")
    def predict() -> Response | tuple[Response, int]:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != REQUIRED_FIELDS:
            return jsonify(error="Provide study_hours and attendance_percentage only."), 400

        study_hours = data["study_hours"]
        attendance = data["attendance_percentage"]
        values = (study_hours, attendance)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            return jsonify(error="Both values must be finite numbers."), 400
        if study_hours < 0 or not 0 <= attendance <= 100:
            return jsonify(
                error="Study hours must be non-negative and attendance must be between 0 and 100."
            ), 400

        prediction = predict_outcome(model, study_hours)
        return jsonify(
            study_hours=study_hours,
            attendance_percentage=attendance,
            prediction=prediction,
        )

    return app


if __name__ == "__main__":
    create_app().run(debug=False)
