from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")
print("Loaded model from student_pass_model.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Student Pass Prediction API",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "study_hours" not in data:
            return jsonify({"error": "Missing study_hours"}), 400
        if "attendance_percentage" not in data:
            return jsonify({"error": "Missing attendance_percentage"}), 400

        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
        student_name = data.get("student_name", "Unknown Student")

        if not (0 <= study_hours <= 24):
            return jsonify({"error": "study_hours must be 0-24"}), 400
        if not (0 <= attendance <= 100):
            return jsonify({"error": "attendance_percentage must be 0-100"}), 400

        features = np.array([[study_hours, attendance]])
        prediction = int(model.predict(features)[0])
        probability = model.predict_proba(features)[0]
        confidence = round(float(max(probability)) * 100, 2)
        label = "Pass" if prediction == 1 else "Fail"

        return jsonify({
            "student_name": student_name,
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": label,
            "confidence": confidence
        }), 200
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input. Numbers required."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)