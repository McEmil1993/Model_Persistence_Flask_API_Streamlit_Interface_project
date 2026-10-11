from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")

@app.route("/")
def home():
    return "Student Pass Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    if study_hours < 0 or not 0 <= attendance <= 100:
        return jsonify({"error": "Invalid values"}), 400

    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)