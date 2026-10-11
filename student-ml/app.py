import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model_attendance.joblib")

app = Flask(__name__)

# Load the retrained model (uses study hours + attendance)
model = joblib.load(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Send JSON body"}), 400

    # 1) Check that both inputs exist
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "study_hours and attendance_percentage are required"}), 400

    # 2) Check that both are numbers
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers"}), 400

    # 3) Validate ranges
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    # 4) Predict
    X = pd.DataFrame({"study_hours": [study_hours], "attendance_percentage": [attendance]})
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "pass" if pred == 1 else "fail",
        "probability_of_passing": round(prob, 3),
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
