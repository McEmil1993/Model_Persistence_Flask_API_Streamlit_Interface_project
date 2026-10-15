"""Train the pass/fail models and save them with joblib."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- Sample data (replace with your course dataset if you have one) ---
data = pd.DataFrame({
    "study_hours":           [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 5, 6, 1, 4, 7, 8],
    "attendance_percentage": [50, 60, 55, 70, 65, 85, 90, 95, 90, 40, 80, 60, 45, 75, 88, 92],
    "passed":                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
})

# Exercise 1: model using study hours only
model_hours = LogisticRegression()
model_hours.fit(data[["study_hours"]], data["passed"])
joblib.dump(model_hours, "student_pass_model.joblib")          # <-- Task 1
print("Saved student_pass_model.joblib")

# Exercise 2: model using study hours + attendance (retrained!)
model_both = LogisticRegression()
model_both.fit(data[["study_hours", "attendance_percentage"]], data["passed"])
joblib.dump(model_both, "student_pass_model_v2.joblib")
print("Saved student_pass_model_v2.joblib")