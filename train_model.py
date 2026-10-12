
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance_percentage": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "passed": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance_percentage"]]
y = df["passed"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")
print("Features:", model.n_features_in_)
print("Saved as student_pass_model.joblib")
