
import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")

test_students = pd.DataFrame({
    "study_hours": [2, 5, 8],
    "attendance_percentage": [80, 80, 80]
})

predictions = model.predict(test_students)

for i, row in test_students.iterrows():
    print(f"Study hours: {row['study_hours']}")
    print(f"Attendance: {row['attendance_percentage']}%")
    print(f"Prediction: {predictions[i]}")
    print("-" * 30)
