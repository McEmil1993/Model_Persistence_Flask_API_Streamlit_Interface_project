from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('student_pass_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check required fields
    if not data or 'study_hours' not in data or 'attendance_percentage' not in data:
        return jsonify({'error': 'Please provide study_hours and attendance_percentage'}), 400

    try:
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance_percentage'])
    except ValueError:
        return jsonify({'error': 'Inputs must be numeric'}), 400

    # Validate attendance between 0 and 100
    if attendance < 0 or attendance > 100:
        return jsonify({'error': 'attendance_percentage must be between 0 and 100'}), 400

    # The current model uses only study_hours.
    # If you retrain with attendance as a second feature, change this to:
    # prediction = model.predict([[study_hours, attendance]])[0]
    prediction = model.predict([[study_hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        'study_hours': study_hours,
        'attendance_percentage': attendance,
        'prediction': result
    })

if __name__ == '__main__':
    app.run(debug=True)
