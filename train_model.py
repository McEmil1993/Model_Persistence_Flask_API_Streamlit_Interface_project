
# Student Pass Prediction Model
from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample training data for demonstration, not real student records.
study_hour_values = range(0, 13)
attendance_values = range(0, 101, 5)

X = []
y = []

for study_hours in study_hour_values:
    for attendance_percentage in attendance_values:
        X.append([study_hours, attendance_percentage])

        # Example rule used to create sample training labels.
        if study_hours >= 6 and attendance_percentage >= 75:
            y.append(1)  # Pass
        else:
            y.append(0)  # Fail

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

model_path = Path(__file__).with_name("student_pass_model.joblib")
joblib.dump(model, model_path)

print(f"Trained model saved as {model_path.name}")

# Load the saved model and test three predictions.
loaded_model = joblib.load(model_path)

print("\nThree sample predictions:")

sample_inputs = [
    [2, 60],
    [7, 60],
    [7, 90]
]

for study_hours, attendance_percentage in sample_inputs:
    prediction_value = int(
        loaded_model.predict([[study_hours, attendance_percentage]])[0]
    )
    result = "Pass" if prediction_value == 1 else "Fail"

    print(
        f"Study hours: {study_hours}, "
        f"Attendance: {attendance_percentage}% -> {result}"
    )

print("\nBoth study hours and attendance are used by the model.")
print("Note: Predictions use illustrative sample training data.")
