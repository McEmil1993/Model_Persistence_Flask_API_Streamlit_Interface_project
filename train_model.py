import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    "study_hours":           [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 5, 6],
    "attendance_percentage": [50, 60, 55, 70, 75, 85, 90, 95, 90, 40, 60, 95],
    "passed":                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
})

# Model 1: study hours only (Exercise 1)
model = LogisticRegression()
model.fit(data[["study_hours"]], data["passed"])
joblib.dump(model, "student_pass_model.joblib")
print("Saved student_pass_model.joblib")

# Model 2: study hours + attendance (Exercise 2)
model2 = LogisticRegression()
model2.fit(data[["study_hours", "attendance_percentage"]], data["passed"])
joblib.dump(model2, "student_pass_model_v2.joblib")
print("Saved student_pass_model_v2.joblib")