import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Sample training data (hours studied, attendance %, passed? 1=yes 0=no) ----
data = pd.DataFrame({
    "study_hours":          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1.5, 2.5, 3.5, 5.5, 6.5, 8.5],
    "attendance_percentage": [40, 50, 55, 60, 70, 75, 85, 90, 95, 100, 90, 45, 80, 60, 95, 70],
    "passed":               [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
})

# ======== Model A: study hours only (Exercise 1 and Exercise 3) ========
model_hours = LogisticRegression()
model_hours.fit(data[["study_hours"]], data["passed"])

# Exercise 1: save with a DIFFERENT filename
joblib.dump(model_hours, os.path.join(BASE_DIR, "student_pass_model.joblib"))
print("Saved: student_pass_model.joblib")

# ======== Model B: study hours + attendance (Exercise 2) ========
model_both = LogisticRegression()
model_both.fit(data[["study_hours", "attendance_percentage"]], data["passed"])
joblib.dump(model_both, os.path.join(BASE_DIR, "student_pass_model_attendance.joblib"))
print("Saved: student_pass_model_attendance.joblib")
