# predict_test.py
import joblib

model = joblib.load("student_pass_model.joblib")
print("✅ Model loaded successfully!\n")

study_hours_list = [2, 5, 8]

for hours in study_hours_list:
    prediction = model.predict([[hours]])[0]
    probability = model.predict_proba([[hours]])[0][1]

    result = "PASS ✅" if prediction == 1 else "FAIL ❌"
    print(f"Study Hours: {hours}  →  Prediction: {result}  "
          f"(Pass probability: {probability:.2%})")