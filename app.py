"""Exercise 2: expose student prediction through a Flask endpoint."""
from flask import Flask, request, jsonify
from model_utils import get_prediction

server = Flask(__name__)


@server.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "POST /predict",
        "required_fields": ["study_hours", "attendance_percentage"],
    })


@server.route("/predict", methods=["POST"])
def predict_result():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing = [
        key for key in ("study_hours", "attendance_percentage")
        if key not in payload
    ]
    if missing:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    try:
        hours = float(payload["study_hours"])
        attendance = float(payload["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({"error": "Both inputs must be numeric."}), 400

    if hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if attendance < 0 or attendance > 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    prediction = get_prediction(hours, attendance)
    response = {
        "study_hours": hours,
        "attendance_percentage": attendance,
        **prediction,
    }
    return jsonify(response), 200


if __name__ == "__main__":
    server.run(host="127.0.0.1", port=5000, debug=False)
