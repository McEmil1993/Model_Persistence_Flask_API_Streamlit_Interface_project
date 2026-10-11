
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Student Prediction Project</h1>
    <p>Flask API is running successfully!</p>
    <p>Use POST /predict to predict student results.</p>
    """

# Load the trained model
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Please provide JSON data"}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Enter valid study hours and attendance"
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    if study_hours < 0:
        return jsonify({
            "error": "Study hours cannot be negative"
        }), 400

    # The current model uses study hours only
    prediction = model.predict([[study_hours]])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)
