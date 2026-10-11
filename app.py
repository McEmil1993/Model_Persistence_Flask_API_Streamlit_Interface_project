from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "student_pass_model.joblib")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Prediction API is running",
        "endpoint": "POST /predict",
        "required_json": {
            "study_hours": 5,
            "attendance_percentage": 90
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric study_hours and attendance_percentage."
        }), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    # This example model uses study hours only.
    prediction = int(model.predict([[study_hours]])[0])
    label = "Likely to Pass" if prediction == 1 else "Needs Improvement"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction_class": prediction,
        "prediction": label,
        "note": "Attendance is validated and returned, but does not affect this model's prediction."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
