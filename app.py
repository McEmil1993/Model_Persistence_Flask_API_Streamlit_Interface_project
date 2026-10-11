from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please send valid JSON."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide study_hours and attendance_percentage."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24."
        }), 400

    # The current model uses study hours only
    input_data = pd.DataFrame({
        "study_hours": [study_hours]
    })

    prediction = model.predict(input_data)[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result,
        "note": "Attendance is validated but does not affect the prediction."
    })


if __name__ == "__main__":
    app.run(debug=True)