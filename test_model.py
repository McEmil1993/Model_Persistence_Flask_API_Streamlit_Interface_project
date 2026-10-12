import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

print("Saved model loaded successfully!\n")

# Make three predictions
study_hours = [2, 5, 8]

for hours in study_hours:
    prediction = model.predict([[hours]])[0]

    print(f"Study Hours: {hours}")
    print(f"Predicted Result: {prediction}")
    print()