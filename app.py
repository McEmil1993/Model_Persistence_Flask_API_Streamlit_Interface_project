from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("student_pass_model.joblib")


@app.route("/")
def home():
    return """
    <h1>Student Performance Prediction API</h1>
    <p>Flask API is running successfully!</p>
    <p>Use POST /predict to make a prediction.</p>
    """


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Please send JSON data."
        }), 400

    if "study_hours" not in data:
        return jsonify({
            "error": "study_hours is required."
        }), 400

    if "attendance_percentage" not in data:
        return jsonify({
            "error": "attendance_percentage is required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])

    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numbers."
        }), 400

    # Validate attendance
    if attendance < 0 or attendance > 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    # Make prediction
    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "note": "The current model uses study_hours only."
    })


if __name__ == "__main__":
    app.run(debug=True)