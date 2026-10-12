from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_pass_model_v2.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Send a JSON body."}), 400

    # Check that both fields exist
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "study_hours and attendance_percentage are required."}), 400

    # Check that both are numbers
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers."}), 400

    # Validate ranges
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction = int(model.predict([[study_hours, attendance]])[0])
    probability = float(model.predict_proba([[study_hours, attendance]])[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "PASS" if prediction == 1 else "FAIL",
        "pass_probability": round(probability, 3),
    })


if __name__ == "__main__":
    app.run(debug=True)