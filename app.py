
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "JSON body is required."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "study_hours and attendance_percentage are required."}), 400

    hours = float(data["study_hours"])
    attendance = float(data["attendance_percentage"])

    if attendance < 0 or attendance > 100:
        return jsonify({"error": "Attendance must be between 0 and 100."}), 400

    result = model.predict([[hours]])
    prediction = "Pass" if result[0] == 1 else "Fail"

    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
