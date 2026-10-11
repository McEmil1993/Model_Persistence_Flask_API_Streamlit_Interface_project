import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Test different study hours
study_hours = [[2], [5], [8]]

# Make predictions
predictions = model.predict(study_hours)

# Display results
for hours, prediction in zip(study_hours, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours[0]} hours")
    print(f"Prediction: {result}")
    print("-" * 25)
