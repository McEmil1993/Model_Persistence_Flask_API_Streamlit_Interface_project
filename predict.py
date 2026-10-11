
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Make three predictions using different study hours
study_hours = [[2], [5], [8]]

for hours in study_hours:
    prediction = model.predict([hours])[0]

    result = "Pass" if prediction == 1 else "Fail"

    print(f"Study Hours: {hours[0]} | Prediction: {result}")
