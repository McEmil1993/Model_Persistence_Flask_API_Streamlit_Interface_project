
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Make three predictions
study_hours_list = [2, 5, 8]

for hours in study_hours_list:
    prediction = model.predict([[hours]])[0]

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(f"Study Hours: {hours} | Prediction: {result}")
