"""Exercise 2 - retrain the model with TWO features:
study_hours and attendance_percentage."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"

data = pd.DataFrame({
    "study_hours":           [0.5, 1, 1.5, 2, 2.5, 3,  3.5, 4,  4.5, 5,  5.5, 6,  7,  8,  2,  3,  1,  4],
    "attendance_percentage": [40, 50, 55, 60, 65, 70,  60, 75, 70, 80,  85, 90, 95, 98, 90, 95, 85, 55],
    "passed":                [0,  0,  0,  0,  0,  0,   0,  1,  1,  1,   1,  1,  1,  1,  1,  1,  0,  0],
})
X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
print(f"Training accuracy: {model.score(X, y):.2f}")

joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")