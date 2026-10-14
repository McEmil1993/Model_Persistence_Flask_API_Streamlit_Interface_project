from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_pass_model_v2.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    if study_hours is None or attendance is None:
        return jsonify({"error": "study_hours and attendance_percentage are required"}), 400

    try:
        study_hours = float(study_hours)
        attendance = float(attendance)
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers"}), 400

    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    prediction = int(model.predict([[study_hours, attendance]])[0])
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
    })

if __name__ == "__main__":
    app.run(debug=True)

# Test with curl (Git Bash):
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"study_hours": 6, "attendance_percentage": 80}'
# Test with curl.exe (PowerShell):
# curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6, \"attendance_percentage\": 80}"