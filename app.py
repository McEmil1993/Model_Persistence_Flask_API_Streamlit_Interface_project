
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("student_pass_model.joblib")

@app.route("/")
def home():
    return "Student Pass Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "study_hours" not in data:
        return jsonify({"error": "Study hours are required"}), 400

    if "attendance" not in data:
        return jsonify({"error": "Attendance is required"}), 400

    study_hours = data["study_hours"]
    attendance = data["attendance"]

    if not isinstance(study_hours, (int, float)) or study_hours < 0:
        return jsonify({"error": "Study hours must be a non-negative number"}), 400

    if not isinstance(attendance, (int, float)) or not 0 <= attendance <= 100:
        return jsonify({"error": "Attendance must be between 0 and 100"}), 400

    prediction = model.predict([[study_hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance": attendance,
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)