import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    "study_hours":           [1, 2, 3, 4, 5, 6, 7, 8, 2, 6],
    "attendance_percentage": [50, 60, 55, 70, 80, 85, 90, 95, 95, 40],
    "passed":                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
})

X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

model = LogisticRegression().fit(X, y)
joblib.dump(model, "student_pass_model_v2.joblib")