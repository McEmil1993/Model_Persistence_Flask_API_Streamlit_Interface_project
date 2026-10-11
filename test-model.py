
import joblib

# Load the saved model
model = joblib.load("student_pass_model.joblib")

# Make three predictions using different study hours
test_students = [
    [2, 60],
    [5, 80],
    [8, 95]
]

for student in test_students:
    study_hours = student[0]
    attendance = student[1]

    prediction = model.predict([student])[0]

    result = "Pass" if prediction == 1 else "Fail"

    print(
        f"Study Hours: {study_hours}, "
        f"Attendance: {attendance}%, "
        f"Prediction: {result}"
    )
