from flask import Flask, request, jsonify
from train_model import load_model

app = Flask(__name__)
model = load_model()

@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass/Fail Prediction API",
        "instructions": "Send a POST request to /predict with study_hours and attendance_percentage."
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400

    if "study_hours" not in data:
        return jsonify({"error": "study_hours is required."}), 400
    if "attendance_percentage" not in data:
        return jsonify({"error": "attendance_percentage is required."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers."}), 400

    if study_hours < 0 or study_hours > 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400
    if attendance < 0 or attendance > 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction = int(model.predict([[study_hours]])[0])
    label = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": label,
        "note": "This demonstration model uses study hours only. Attendance is returned but does not affect the prediction."
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
