from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running!"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please send JSON data."}), 400

    hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        return jsonify({"error": "Study hours must be a number."}), 400

    if not 0 <= hours <= 24:
        return jsonify({"error": "Study hours must be between 0 and 24."}), 400

    if isinstance(attendance, bool) or not isinstance(attendance, (int, float)):
        return jsonify({"error": "Attendance must be a number."}), 400

    if not 0 <= attendance <= 100:
        return jsonify({"error": "Attendance must be between 0 and 100."}), 400

    # Temporary rule only; replace with the trained model later.
    prediction = "Pass" if hours >= 3.5 else "Fail"

    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
