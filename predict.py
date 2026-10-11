
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Test three different study hours
for hours in [[2], [5], [8]]:
    prediction = model.predict([hours])[0]

    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours[0]} - Prediction: {result}")