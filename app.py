
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please provide valid JSON."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Enter numeric study_hours and attendance_percentage."
        }), 400

    if study_hours < 0:
        return jsonify({"error": "Study hours cannot be negative."}), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "note": "The model uses study hours only."
    })


if __name__ == "__main__":
    app.run(debug=True)
