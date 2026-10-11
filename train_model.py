"""Exercise 1 (+ model for Exercise 2): train, save, load, and predict."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---- Training data (sample) ----
data = pd.DataFrame({
    "study_hours":           [1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 1, 4, 6, 3],
    "attendance_percentage": [40, 55, 70, 50, 65, 60, 75, 85, 80, 90, 95, 98, 90, 45, 55, 95],
    "passed":                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
})

# ---- Model A: study hours only (Exercise 1 & 3) ----
model_hours = LogisticRegression()
model_hours.fit(data[["study_hours"]], data["passed"])
joblib.dump(model_hours, "student_pass_model.joblib")          # SAVE

# ---- Model B: study hours + attendance (Exercise 2) ----
model_both = LogisticRegression()
model_both.fit(data[["study_hours", "attendance_percentage"]], data["passed"])
joblib.dump(model_both, "student_pass_model_attendance.joblib")

print("Models saved.")

# ---- Exercise 1: LOAD and make three predictions ----
loaded = joblib.load("student_pass_model.joblib")               # LOAD
for hours in [1, 4, 8]:
    pred = loaded.predict(pd.DataFrame({"study_hours": [hours]}))[0]
    print(f"{hours} study hours -> {'Pass' if pred == 1 else 'Fail'}")

# Why no retraining? The learned parameters (coefficients and intercept)
# are stored inside the .joblib file. Loading restores the fitted model,
# so predict() only applies those saved numbers to the new input.
