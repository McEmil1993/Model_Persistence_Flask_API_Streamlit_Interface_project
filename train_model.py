import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    "study_hours":           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance_percentage": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "pass":                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})

X = data[["study_hours", "attendance_percentage"]].values   # <-- add .values
y = data["pass"].values                                      # <-- add .values

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)