
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Test three different study-hour values
study_hours = [[2], [5], [8]]

# Generate predictions
predictions = model.predict(study_hours)

# Display the results
for hours, prediction in zip(study_hours, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours[0]}")
    print(f"Prediction: {result}")
    print()
