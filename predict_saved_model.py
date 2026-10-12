import joblib

# Load the saved model (no training happens here)
model = joblib.load("student_pass_model.joblib")

for hours in [2, 5.5, 8]:
    prediction = model.predict([[hours]])[0]
    probability = model.predict_proba([[hours]])[0][1]
    result = "PASS" if prediction == 1 else "FAIL"
    print(f"Study hours: {hours} -> {result} (pass probability: {probability:.2f})")