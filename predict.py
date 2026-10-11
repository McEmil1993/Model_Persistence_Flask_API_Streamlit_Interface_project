
import joblib
from pathlib import Path

# Load the saved model
model_path = Path(__file__).parent / "student_pass_model.joblib"
model = joblib.load(model_path)

print("Student Pass Prediction System")
print("--------------------------------")

# Get student information
study_hours = float(input("Enter study hours per day: "))
attendance = float(input("Enter attendance percentage (0-100): "))

# Validate inputs
if study_hours < 0:
    print("Error: Study hours cannot be negative.")

elif not 0 <= attendance <= 100:
    print("Error: Attendance must be between 0 and 100.")

else:
    # Predict using both features
    student_data = [[study_hours, attendance]]
    prediction = model.predict(student_data)[0]

    if prediction == 1:
        print("Prediction: PASS")
    else:
        print("Prediction: FAIL")
