
import joblib

# Load the saved machine learning model
model = joblib.load("student_pass_model.joblib")

# Three different study-hour examples
study_hours = [[2], [5], [8]]

# Make predictions
for hours in study_hours:
    prediction = model.predict([hours])

    if prediction[0] == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(f"Study hours: {hours[0]} - Prediction: {result}")

