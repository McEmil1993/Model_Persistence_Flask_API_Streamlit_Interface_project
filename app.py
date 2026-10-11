"""Exercise 2 - Flask API accepting study_hours and attendance_percentage."""
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.get_json(silent=True) or request.values   # JSON body or query/form
    try:
        study_hours = float(data.get("study_hours"))
        attendance = float(data.get("attendance_percentage"))
    except (TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage are required numbers"), 400

    if study_hours < 0:
        return jsonify(error="study_hours must be 0 or more"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    features = [[study_hours, attendance]]
    result = int(model.predict(features)[0])
    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="pass" if result == 1 else "fail",
        pass_probability=round(float(model.predict_proba(features)[0][1]), 3),
    )


if __name__ == "__main__":
    app.run(debug=True)

# Test with curl.exe (Windows):
#   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 5, \"attendance_percentage\": 85}"
#   curl.exe "http://127.0.0.1:5000/predict?study_hours=5&attendance_percentage=120"   # -> 400 error
