
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Test three different study-hour values
test_hours = [[2], [5], [8]]

# Make predictions using the loaded model
for hours in test_hours:
    prediction = model.predict([hours])[0]

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(f"Study hours: {hours[0]} - Prediction: {result}")
