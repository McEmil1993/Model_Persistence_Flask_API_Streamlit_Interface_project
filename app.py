from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "student_pass_model.joblib")
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass/Fail Prediction API",
        "endpoint": "POST /predict",
        "example_input": {
            "study_hours": 5,
            "attendance_percentage": 90
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400

    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    if isinstance(study_hours, bool) or not isinstance(study_hours, (int, float)):
        return jsonify({"error": "study_hours must be a number."}), 400
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400

    if isinstance(attendance, bool) or not isinstance(attendance, (int, float)):
        return jsonify({"error": "attendance_percentage must be a number from 0 to 100."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction_value = int(model.predict([[float(study_hours)]])[0])
    prediction_text = "Pass" if prediction_value == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction_text,
        "note": "This model uses study_hours only; attendance is validated and returned but does not affect the prediction."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)