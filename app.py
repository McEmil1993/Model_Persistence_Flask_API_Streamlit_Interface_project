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
    
    # Check if all required inputs exist
    required_keys = ['study_hours', 'previous_grade', 'attendance_percentage']
    if not data or not all(k in data for k in required_keys):
        return jsonify({'error': f'Please provide all required fields: {required_keys}'}), 400

    try:
        study_hours = float(data['study_hours'])
        prev_grade = float(data['previous_grade'])
        attendance = float(data['attendance_percentage'])
    except ValueError:
        return jsonify({'error': 'Invalid input types. All values must be numbers.'}), 400

    # Validate ranges (Grade: 0-100, Attendance: 0-100)
    if not (0 <= prev_grade <= 100) or not (0 <= attendance <= 100):
        return jsonify({'error': 'Validation Error: previous_grade and attendance_percentage must be between 0 and 100.'}), 400

    # Make prediction
    features = [[study_hours, prev_grade, attendance]]
    prediction = int(model.predict(features)[0])
    result_status = "Pass" if prediction == 1 else "Fail"

    # Return structured JSON response
    return jsonify({
        'study_hours': study_hours,
        'previous_grade': prev_grade,
        'attendance_percentage': attendance,
        'prediction_code': prediction,
        'prediction_result': result_status
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)