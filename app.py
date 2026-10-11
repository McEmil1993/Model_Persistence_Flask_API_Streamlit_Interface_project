from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")

@app.route("/")
def home():
    return jsonify({"message": "API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Check if inputs exist
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "Need study_hours and attendance_percentage"}), 400
    
    study_hours = float(data["study_hours"])
    attendance = float(data["attendance_percentage"])
    
    # Check attendance range
    if attendance < 0 or attendance > 100:
        return jsonify({"error": "Attendance must be 0-100"}), 400
    
    # Predict
    prediction = int(model.predict([[study_hours]])[0])
    
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "result": "PASS" if prediction == 1 else "FAIL"
    })

if __name__ == "__main__":
    app.run(debug=True)