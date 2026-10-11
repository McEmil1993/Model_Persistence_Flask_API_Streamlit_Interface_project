"""Exercise 2 - Flask API with study_hours + attendance_percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/")
def home():
    return jsonify(
        message="Student Pass Predictor API",
        usage='POST /predict with JSON: {"study_hours": 7, "attendance_percentage": 85}',
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Accept a JSON body (POST) or query string (GET)
    data = request.get_json(silent=True)
    source = data if isinstance(data, dict) else request.args
    print(f"[DEBUG] {request.method} Content-Type={request.content_type} data={dict(source)}")

    # Check that both fields were sent
    missing = [f for f in ("study_hours", "attendance_percentage") if f not in source]
    if missing:
        hint = ""
        if request.method == "POST" and data is None:
            hint = " (no valid JSON body found: set Body to raw > JSON and check your quotes/commas)"
        return jsonify(error=f"Missing field(s): {', '.join(missing)}{hint}"), 400

    # Check that both are numbers
    try:
        study_hours = float(source["study_hours"])
        attendance = float(source["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage must be numbers"), 400

    if study_hours < 0:
        return jsonify(error="study_hours cannot be negative"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    features = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["study_hours", "attendance_percentage"],
    )
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    print(f"[DEBUG] prediction={prediction} probability={probability:.3f}")

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="pass" if prediction == 1 else "fail",
        pass_probability=round(probability, 3),
    )


if __name__ == "__main__":
    # use_reloader=False stops Flask restarting when unrelated packages (e.g. streamlit) change
    app.run(debug=True, use_reloader=False)