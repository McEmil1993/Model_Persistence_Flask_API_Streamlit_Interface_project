"""
Exercise 2 (required note) - Retrain the model using BOTH study hours and
attendance percentage so attendance really influences the prediction.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model_attendance.joblib"

data = pd.DataFrame({
    "study_hours":           [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 2,  4,  6,  8,  3,  5,  7,  9],
    "attendance_percentage": [40, 55, 60, 70, 65, 85, 75, 90, 95, 98, 90, 50, 60, 80, 85, 95, 55, 60],
    "passed":                [0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  1,  0,  1,  0,  1],
})
X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, MODEL_FILE)
print(f"Model trained with 2 features and saved as '{MODEL_FILE}'")
print("Training accuracy:", round(model.score(X, y), 2))
