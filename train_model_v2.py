import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    "study_hours":           [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 1, 3, 4, 2, 10, 5],
    "attendance_percentage": [50, 60, 90, 55, 80, 60, 85, 65, 90, 70, 50, 60, 85, 55, 95, 70, 60, 95, 95, 95, 40, 90, 75],
    "passed":                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
})

X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "student_pass_model_v2.joblib")
print("Model saved to student_pass_model_v2.joblib")