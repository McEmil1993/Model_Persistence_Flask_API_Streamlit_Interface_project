
from flask import Flask, request, jsonify
from load_model import load_saved_model, predict_student

app = Flask(__name__)

# Load the existing model once when the API starts
model = load_saved_model()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Caitom Student Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "A valid JSON object is required"}), 400

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric study_hours and attendance_percentage"
        }), 400

    if hours < 0:
        return jsonify({
            "error": "Study hours cannot be negative"
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    result = predict_student(model, hours)

    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
