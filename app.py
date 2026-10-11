"""Exercise 2 — Flask API with study_hours + attendance_percentage."""
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model trained in Exercise 1 (uses study_hours + attendance_percentage)
model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    # 1. Accept both inputs
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Missing fields. Required: 'study_hours' and 'attendance_percentage'"
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers"}), 400

    # 2. Validate attendance is between 0 and 100
    if not (0 <= attendance <= 100):
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100"
        }), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be >= 0"}), 400

    # 3. Predict with BOTH features (model retrained on [hours, attendance]).
    prediction = int(model.predict([[study_hours, attendance]])[0])
    result = "PASS" if prediction == 1 else "FAIL"
    label = "Pass" if prediction == 1 else "Fail"
    try:
        proba = model.predict_proba([[study_hours, attendance]])[0]
        confidence = round(float(max(proba)) * 100, 1)
    except Exception:
        confidence = None
    student_name = data.get("student_name", "")
    if student_name is None:
        student_name = ""

    # 4. Return both values + prediction in JSON
    return jsonify({
        "student_name": student_name,
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "prediction_label": label,
        "result": result,
        "confidence": confidence,
        "note": "both study_hours and attendance_percentage shape the prediction"
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass API. POST JSON to /predict",
        "example": {"student_name": "Alex", "study_hours": 8, "attendance_percentage": 85},
        "returns": ["student_name", "study_hours", "attendance_percentage",
                    "prediction", "prediction_label", "result", "confidence", "note"]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)

# Retraining note (done — model now uses 2 features):
#   X_train = [[hours, attendance], ...] labeled by the pass rule in train_model.py
#   model.fit(X_train, y_train)
#   joblib.dump(model, "student_pass_model.joblib")
# Prediction uses both: model.predict([[study_hours, attendance]])
