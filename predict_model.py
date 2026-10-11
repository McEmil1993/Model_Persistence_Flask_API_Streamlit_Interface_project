
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Make three predictions
study_hours = [[2], [5], [8]]
predictions = model.predict(study_hours)

for hours, prediction in zip(study_hours, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours[0]}, Result: {result}")
