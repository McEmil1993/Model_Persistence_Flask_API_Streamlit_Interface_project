"""Exercise 1 - Model persistence (plus the attendance model for Exercise 2)."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------
# Training data (study hours, attendance %, passed? 1 = pass)
# ---------------------------------------------------------------
data = pd.DataFrame(
    {
        "study_hours":          [1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10, 4, 6, 2],
        "attendance_percentage": [40, 55, 70, 50, 65, 60, 75, 90, 70, 85, 90, 95, 100, 95, 55, 90],
        "passed":               [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    }
)

# ---------------------------------------------------------------
# Model 1: study hours only  ->  student_pass_model.joblib
# ---------------------------------------------------------------
X_hours = data[["study_hours"]]
y = data["passed"]

hours_model = LogisticRegression()
hours_model.fit(X_hours, y)

joblib.dump(hours_model, "student_pass_model.joblib")          # SAVE
print("Saved: student_pass_model.joblib")

# ---------------------------------------------------------------
# Model 2: study hours + attendance -> student_pass_model_attendance.joblib
# (used by app.py so attendance really influences the prediction)
# ---------------------------------------------------------------
X_both = data[["study_hours", "attendance_percentage"]]

both_model = LogisticRegression()
both_model.fit(X_both, y)

joblib.dump(both_model, "student_pass_model_attendance.joblib")
print("Saved: student_pass_model_attendance.joblib")

# ---------------------------------------------------------------
# Load the saved model and make three predictions (NO retraining)
# ---------------------------------------------------------------
loaded_model = joblib.load("student_pass_model.joblib")        # LOAD

for hours in [1, 4, 8]:
    sample = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(sample)[0]
    prob = loaded_model.predict_proba(sample)[0][1]
    label = "PASS" if pred == 1 else "FAIL"
    print(f"Study hours = {hours:>2} -> {label} (pass probability {prob:.2f})")

# Why no retraining is needed:
# Training (fit) learns the model's parameters (coefficient and intercept)
# from the data. joblib.dump() stores the whole fitted object, including those
# learned parameters. joblib.load() restores it exactly, so predict() only does
# a quick calculation with the stored numbers - the training data is not needed.