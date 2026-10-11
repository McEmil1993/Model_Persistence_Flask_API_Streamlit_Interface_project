import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"

# Sample data: study hours + attendance % -> pass (1) / fail (0)
data = pd.DataFrame({
    "study_hours":           [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 6, 9, 1, 5, 7, 4],
    "attendance_percentage": [50, 60, 55, 70, 75, 85, 90, 95, 90, 95, 60, 80, 40, 65, 70, 85],
    "passed":                [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
})

X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

model = LogisticRegression()
model.fit(X, y)

# Exercise 1: save with a different filename
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# Exercise 1: load it back and make three predictions (no retraining)
loaded = joblib.load(MODEL_FILE)
for hours in (1, 4, 8):
    sample = pd.DataFrame([[hours, 80]], columns=X.columns)
    label = "Pass" if loaded.predict(sample)[0] == 1 else "Fail"
    print(f"{hours} study hours, 80% attendance -> {label}")

print("""
Why no retraining is needed:
Training (fit) finds the model's learned parameters (coefficients and intercept).
joblib.dump() stores those learned values on disk. joblib.load() restores the exact
same fitted model, so predict() just applies the stored values to new input.""")
