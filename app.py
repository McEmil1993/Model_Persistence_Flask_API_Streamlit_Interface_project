
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "JSON data is required"}), 400

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])

        if not 0 <= attendance <= 100:
            return jsonify({
                "error": "Attendance must be between 0 and 100"
            }), 400

        if not 0 <= hours <= 24:
            return jsonify({
                "error": "Study hours must be between 0 and 24"
            }), 400

        prediction = int(model.predict([[hours]])[0])
        result = "Pass" if prediction == 1 else "Fail"

        return jsonify({
            "study_hours": hours,
            "attendance_percentage": attendance,
            "prediction": result
        })

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Enter valid numeric inputs"
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
