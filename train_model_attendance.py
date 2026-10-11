"""Exercise 2 (retraining) - model that uses study hours AND attendance."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model_attendance.joblib"

data = pd.DataFrame({
    "study_hours":           [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 2,  8,  4,  6],
    "attendance_percentage": [50, 60, 55, 70, 65, 80, 75, 90, 95, 98, 95, 60, 90, 70],
    "passed":                [0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1],
})

X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
