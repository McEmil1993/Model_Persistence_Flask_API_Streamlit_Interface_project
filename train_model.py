# Student Pass Prediction Model
from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier

# Illustrative training data generated from combinations of study hours and attendance.
# These are sample labels for demonstrating the exercise, not real student records.
study_hour_values = range(0, 13)
attendance_values = range(0, 101, 10)

X = []
y = []

for study_hours in study_hour_values:
    for attendance_percentage in attendance_values:
        X.append([study_hours, attendance_percentage])

        # Example pattern used to create sample training labels:
        # pass when study hours are at least 6 and attendance is at least 75%.
        result = 1 if study_hours >= 6 and attendance_percentage >= 75 else 0
        y.append(result)

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

model_path = Path(__file__).with_name("student_pass_model.joblib")
joblib.dump(model, model_path)
print(f"Trained two-input model saved as {model_path.name}")

loaded_model = joblib.load(model_path)
print("\nThree sample predictions:")
sample_inputs = [
    (2, 60),
    (7, 60),
    (7, 90),
]

for study_hours, attendance_percentage in sample_inputs:
    prediction_value = int(loaded_model.predict([[study_hours, attendance_percentage]])[0])
    result = "Pass" if prediction_value == 1 else "Fail"
    print(
        f"Study hours: {study_hours}, Attendance: {attendance_percentage}% "
        f"-> Prediction: {result}"
    )

print(
    "\nThe model was saved and loaded with joblib. It now accepts both "
    "study_hours and attendance_percentage as input features."
)
print(
    "NOTE: The training data uses an illustrative sample pattern, not real "
    "student records. Its predictions are only for demonstrating the exercise."
)
