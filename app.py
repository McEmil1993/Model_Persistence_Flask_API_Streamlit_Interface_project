import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")
print("[OK] Model loaded. Flask API is ready.")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "usage": "POST /predict with JSON: study_hours and attendance_percentage"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "Both study_hours and attendance_percentage are required."}), 400
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Inputs must be numeric."}), 400
    if not (0 <= attendance <= 100):
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    prediction = int(model.predict([[study_hours]])[0])
    probability = float(model.predict_proba([[study_hours]])[0][1])
    label = "Pass" if prediction == 1 else "Fail"
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": label,
        "pass_probability": round(probability, 4),
        "note": "The current model was trained only on study hours. Retrain the model to let attendance influence predictions."
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)