from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    # Validate JSON input
    if not isinstance(data, dict):
        return jsonify({"error": "A valid JSON object is required"}), 400

    # Validate required fields
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required"
        }), 400

    study_hours = data["study_hours"]
    attendance = data["attendance_percentage"]

    # Validate numeric inputs
    if isinstance(study_hours, bool) or not isinstance(study_hours, (int, float)):
        return jsonify({"error": "study_hours must be a number"}), 400

    if isinstance(attendance, bool) or not isinstance(attendance, (int, float)):
        return jsonify({"error": "attendance_percentage must be a number"}), 400

    # Validate attendance range
    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100"
        }), 400

    # Validate study hours
    if study_hours < 0:
        return jsonify({
            "error": "study_hours cannot be negative"
        }), 400

    # Make prediction
    pred = model.predict([[study_hours, attendance]])[0]
    label = "Pass" if pred == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": label
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)