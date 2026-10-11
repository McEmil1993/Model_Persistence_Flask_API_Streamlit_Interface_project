import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------
# Sample training data (replace with your lesson's dataset if different)
# ---------------------------------------------------------------
data = pd.DataFrame({
    "study_hours":           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 6, 8, 1, 9],
    "attendance_percentage": [50, 60, 55, 70, 65, 80, 85, 90, 95, 100, 90, 40, 60, 75, 30, 85],
    "passed":                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
})

# ===============================================================
# EXERCISE 1 - Model persistence (study hours only)
# ===============================================================
X_hours = data[["study_hours"]]
y = data["passed"]

model = LogisticRegression()
model.fit(X_hours, y)

# Save the trained model to a file
joblib.dump(model, "student_pass_model.joblib")
print("Saved: student_pass_model.joblib")

# Load it back (no training needed!)
loaded_model = joblib.load("student_pass_model.joblib")

# Three predictions with different study hours
for hours in [2, 5, 9]:
    pred = loaded_model.predict(pd.DataFrame({"study_hours": [hours]}))[0]
    prob = loaded_model.predict_proba(pd.DataFrame({"study_hours": [hours]}))[0][1]
    print(f"Study hours = {hours}: {'Pass' if pred == 1 else 'Fail'} (pass probability {prob:.2f})")

# Why no retraining? Because fit() already stored the learned values
# (coefficients and intercept) inside the model object. joblib.dump()
# saved those learned values, and joblib.load() restores them, so
# predict() only applies the stored formula to the new input.

# ===============================================================
# EXERCISE 2 - Retrain with BOTH features so attendance matters
# ===============================================================
X_both = data[["study_hours", "attendance_percentage"]]
model_v2 = LogisticRegression(max_iter=1000)
model_v2.fit(X_both, y)

joblib.dump(model_v2, "student_pass_model_v2.joblib")
print("Saved: student_pass_model_v2.joblib")