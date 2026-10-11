"""Exercise 2 - retrain the model with study_hours AND attendance_percentage."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model_v2.joblib"

data = pd.DataFrame({
    "study_hours":           [0.5, 1,  1.5, 2,  2.5, 3,  3.5, 4,  4,  5,  6,  7,  8,  2,  3,  1],
    "attendance_percentage": [40,  50, 60,  55, 65,  70, 75,  80, 60, 85, 90, 95, 100, 90, 55, 95],
    "passed":                [0,   0,  0,   0,  0,   0,  1,   1,  0,  1,  1,  1,  1,   1,  0,  0],
})
X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
joblib.dump(model, MODEL_FILE)
print(f"Saved {MODEL_FILE} | training accuracy: {model.score(X, y):.2f}")
