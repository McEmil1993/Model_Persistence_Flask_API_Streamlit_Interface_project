import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample data (replace with your own dataset if you have one)
data = pd.DataFrame({
    "study_hours":           [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 10],
    "attendance_percentage": [40, 85, 55, 90, 50, 75, 60, 95, 45, 70, 90, 55, 80, 95, 50, 75, 60, 90, 65, 70],
    "passed":                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
})

# Model 1: study hours only (Exercises 1 and 3)
model = LogisticRegression()
model.fit(data[["study_hours"]], data["passed"])
joblib.dump(model, "student_pass_model.joblib")

# Model 2: study hours + attendance (Exercise 2)
model_att = LogisticRegression()
model_att.fit(data[["study_hours", "attendance_percentage"]], data["passed"])
joblib.dump(model_att, "student_pass_model_attendance.joblib")

print("Models saved.")