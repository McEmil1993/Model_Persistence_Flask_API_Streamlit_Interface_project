from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")

# Range the model was actually trained on
MIN_HOURS = 1.0
MAX_HOURS = 8.0


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    # --- Validate study_hours presence and type ---
    if study_hours is None:
        return jsonify({"error": "study_hours is required"}), 400
    try:
        study_hours = float(study_hours)
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours must be a number"}), 400

    # --- Validate study_hours range ---
    if study_hours < MIN_HOURS or study_hours > MAX_HOURS:
        return jsonify({
            "error": (
                f"study_hours must be between {MIN_HOURS} and {MAX_HOURS}. "
                f"The model was only trained on this range."
            )
        }), 400

    # --- Validate attendance ---
    if attendance is None or not (0 <= attendance <= 100):
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100"
        }), 400

    # --- Prediction ---
    prediction = model.predict([[study_hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)