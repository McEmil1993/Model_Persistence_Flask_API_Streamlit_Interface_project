
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

print("Model loaded successfully!")

# Make three predictions
study_hours = [[2], [5], [8]]

for hours in study_hours:
    prediction = model.predict([hours])
    print("Study hours:", hours[0])
    print("Prediction:", prediction[0])
    print("----------------")
