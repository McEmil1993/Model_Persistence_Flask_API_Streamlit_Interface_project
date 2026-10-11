"""Exercise 2 (supporting script)
Retrains the model using BOTH study_hours and attendance_percentage, so that
attendance really influences the prediction. Used by app.py.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model_attendance.joblib"

data = pd.DataFrame({
    "study_hours":           [0,  1,  1,  2,  2,  3,  3,  4,  5,  5,  6,  7,  8,  9,  10, 2,  3,  4],
    "attendance_percentage": [40, 50, 85, 60, 90, 55, 80, 65, 70, 95, 75, 85, 90, 95, 100, 95, 90, 50],
    "passed":                [0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
})
X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
print("Training accuracy:", round(model.score(X, y), 2))
