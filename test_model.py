
import joblib

# Load the previously trained model
model = joblib.load("models/student_pass_model.joblib")

# Three new prediction inputs
study_hours = [[2], [5], [7]]

# Make predictions
predictions = model.predict(study_hours)

# Display the results
for hours, prediction in zip(study_hours, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours[0]} | Prediction: {result}")
