from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError, KeyError):
        return jsonify({
            "error": "Provide valid study_hours and attendance_percentage"
        }), 400

    if hours < 0 or hours > 24:
        return jsonify({"error": "Study hours must be 0 to 24"}), 400

    if attendance < 0 or attendance > 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    prediction = model.predict([[hours]])[0]

    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)