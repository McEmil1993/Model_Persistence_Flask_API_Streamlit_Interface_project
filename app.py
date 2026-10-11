from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('student_pass_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        study_hours = data.get('study_hours')
        attendance = data.get('attendance_percentage')

        # Validation
        if study_hours is None or attendance is None:
            return jsonify({"error": "Both study_hours and attendance_percentage required"}), 400

        if not (0 <= attendance <= 100):
            return jsonify({"error": "attendance_percentage must be 0-100"}), 400

        # 🆕 IMPORTANTE: 2 values na ipapasa sa model
        prediction = model.predict([[study_hours, attendance]])[0]
        result = "Pass" if prediction == 1 else "Fail"

        return jsonify({
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)