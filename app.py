from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'student_pass_model.joblib'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not found. Run train_model.py first.'}), 500

    data = request.get_json()
    
    # Check if inputs exist
    if not data or 'study_hours' not in data or 'attendance_percentage' not in data:
        return jsonify({'error': 'Please provide both "study_hours" and "attendance_percentage".'}), 400

    try:
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance_percentage'])
    except ValueError:
        return jsonify({'error': 'Invalid input types. Values must be numbers.'}), 400

    # Validate that attendance is between 0 and 100
    if not (0 <= attendance <= 100):
        return jsonify({'error': 'Validation Error: attendance_percentage must be between 0 and 100.'}), 400

    # Make prediction
    features = [[study_hours, attendance]]
    prediction = int(model.predict(features)[0])
    result_status = "Pass" if prediction == 1 else "Fail"

    # Return values and model prediction in JSON
    return jsonify({
        'study_hours': study_hours,
        'attendance_percentage': attendance,
        'prediction_code': prediction,
        'prediction_result': result_status
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)