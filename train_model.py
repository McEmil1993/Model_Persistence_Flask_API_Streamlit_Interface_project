import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["study_hours", "attendance_percentage"]

data = pd.DataFrame({
    "study_hours":           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    "attendance_percentage": [60, 55, 70, 75, 80, 85, 90, 88, 95, 98, 50, 65, 72, 78, 82, 92],
    "passed":                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
})

X = data[FEATURES]
y = data["passed"]

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")

loaded = joblib.load("student_pass_model.joblib")

tests = pd.DataFrame([[1, 50], [5, 80], [9, 95]], columns=FEATURES)
for (_, row), pred in zip(tests.iterrows(), loaded.predict(tests)):
    print(f"{row.study_hours} hrs, {row.attendance_percentage}% attendance -> "
          f"{'Pass' if pred == 1 else 'Fail'}")
