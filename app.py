from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    hours = data.get("study_hours")
    att = data.get("attendance_percentage")

    if hours is None or att is None:
        return jsonify(error="Both study_hours and attendance_percentage are required"), 400
    if not isinstance(hours, (int, float)) or not isinstance(att, (int, float)):
        return jsonify(error="Inputs must be numbers"), 400
    if not 0 <= att <= 100:
        return jsonify(error="Attendance must be between 0 and 100"), 400

    pred = int(model.predict([[hours, att]])[0])
    return jsonify(study_hours=hours,
                   attendance_percentage=att,
                   prediction=pred)


if __name__ == "__main__":
    app.run(debug=True)
