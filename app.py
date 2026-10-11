from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('student_prediction_project/student_pass_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'study_hours' not in data or 'attendance_percentage' not in data:
        return jsonify({"error": "Missing inputs"}), 400
    try:
        hours = float(data['study_hours'])
        att = float(data['attendance_percentage'])
    except ValueError:
        return jsonify({"error": "Inputs must be numeric"}), 400
    if att < 0 or att > 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400
    if hours < 0 or hours > 24:
        return jsonify({"error": "study_hours must be between 0 and 24"}), 400
    pred = int(model.predict([[hours, att]])[0])
    prob = float(model.predict_proba([[hours, att]])[0][1])
    return jsonify({
        "status": "success",
        "study_hours": hours,
        "attendance_percentage": att,
        "prediction": "Pass" if pred == 1 else "Fail",
        "pass_probability": round(prob, 4)
    }), 200

if __name__ == '__main__':
    app.run(port=5000)
