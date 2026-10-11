"""Exercise 1 - train the models and save them with joblib."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Small example dataset (hours studied, attendance %, passed? 1 = yes, 0 = no)
data = pd.DataFrame({
    "study_hours":           [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
    "attendance_percentage": [40, 55, 50, 60, 45, 70, 65, 62, 80, 75, 85, 90, 70, 95, 88, 92],
    "passed":                [0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1],
})

# ---- Model 1: study hours only (Exercise 1 and 3) ----
model_hours = LogisticRegression()
model_hours.fit(data[["study_hours"]], data["passed"])
joblib.dump(model_hours, "student_pass_model.joblib")      # <-- new filename

# ---- Model 2: study hours + attendance (Exercise 2) ----
model_both = LogisticRegression()
model_both.fit(data[["study_hours", "attendance_percentage"]], data["passed"])
joblib.dump(model_both, "student_pass_model_v2.joblib")

print("Saved: student_pass_model.joblib and student_pass_model_v2.joblib")
