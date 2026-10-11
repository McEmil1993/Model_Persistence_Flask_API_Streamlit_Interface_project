# Exercise 2 - Flask API using BOTH study hours and attendance percentage
import os

import joblib
from flask import Flask, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "student_pass_model_v2.joblib")


def create_app():
    app = Flask(__name__)
    model = joblib.load(MODEL_PATH)  # trained on study hours AND attendance

    def bad_request(message):
        return jsonify({"error": message}), 400

    @app.get("/")
    def home():
        return jsonify({"message": "Student Pass Prediction API is running"})

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        # JSON body for POST, query string for GET
        if request.method == "POST":
            data = request.get_json(silent=True)
        else:
            data = request.args.to_dict()

        if not isinstance(data, dict) or len(data) == 0:
            return bad_request("Send study_hours and attendance_percentage")

        # 1. The API must accept both inputs
        for field in ("study_hours", "attendance_percentage"):
            if field not in data:
                return bad_request(f"{field} is required")

        try:
            hours = float(data["study_hours"])
            attendance = float(data["attendance_percentage"])
        except (TypeError, ValueError):
            return bad_request("Both values must be numbers")

        if hours < 0:
            return bad_request("study_hours cannot be negative")

        # 2. Attendance must be between 0 and 100
        if not (0 <= attendance <= 100):
            return bad_request("attendance_percentage must be between 0 and 100")

        # 3. Return both values and the prediction as JSON
        label = model.predict([[hours, attendance]])[0]
        return jsonify({
            "study_hours": hours,
            "attendance_percentage": attendance,
            "prediction": "Pass" if label == 1 else "Fail",
        })

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)