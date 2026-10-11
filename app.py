from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model once at startup
model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both 'study_hours' and 'attendance_percentage' are required"
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numbers"
        }), 400

    # Validation
    if not (0 <= attendance <= 100):
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400

    # Prediction using BOTH features
    prediction = model.predict([[study_hours, attendance]])[0]
    proba = model.predict_proba([[study_hours, attendance]])[0][1]
    label = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": label,
        "pass_probability": round(float(proba), 4)
    })

if __name__ == "__main__":
    app.run(debug=True)