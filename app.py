from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load saved model
model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    study_hours = data.get("study_hours")
    attendance_percentage = data.get("attendance_percentage")

    # Validation
    if attendance_percentage < 0 or attendance_percentage > 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100"
        }), 400

    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)