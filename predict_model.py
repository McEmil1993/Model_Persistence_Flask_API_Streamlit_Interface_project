import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Test with three different study hours
study_hours = [[2], [5], [8]]

# Make predictions
predictions = model.predict(study_hours)

# Display the results
for hours, prediction in zip(study_hours, predictions):
    print(f"Study hours: {hours[0]}, Prediction: {prediction}")