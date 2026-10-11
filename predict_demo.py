"""Exercise 1 - Load the saved model and make three predictions WITHOUT retraining."""
import joblib

model = joblib.load("student_pass_model.joblib")

for hours in (2, 5, 8):  # three different study-hour values
    pred = model.predict([[hours]])[0]
    prob = model.predict_proba([[hours]])[0][1]
    label = "PASS" if pred == 1 else "FAIL"
    print(f"Study hours = {hours}: prediction = {label} (pass probability = {prob:.2f})")
