import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Training data (1 = Pass, 0 = Fail)
data = pd.DataFrame({
    "study_hours":           [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8],
    "attendance_percentage": [40, 50, 55, 60, 65, 70, 60, 85, 80, 90, 95, 100],
    "passed":                [0,   0, 0,   0, 0,   0, 1,   1, 1,  1,  1,  1],
})

# 1. Train and SAVE the study-hours-only model
model = LogisticRegression()
model.fit(data[["study_hours"]], data["passed"])
joblib.dump(model, "student_pass_model.joblib")
print("Saved student_pass_model.joblib")

# 2. Train and SAVE a model that also uses attendance (needed for Exercise 2)
model_att = LogisticRegression()
model_att.fit(data[["study_hours", "attendance_percentage"]], data["passed"])
joblib.dump(model_att, "student_pass_model_attendance.joblib")
print("Saved student_pass_model_attendance.joblib")

# 3. LOAD the saved model and make three predictions (no .fit() here)
loaded_model = joblib.load("student_pass_model.joblib")

for hours in [1, 3.5, 6]:
    X = pd.DataFrame({"study_hours": [hours]})
    result = loaded_model.predict(X)[0]
    print(f"Study hours = {hours}: {'Pass' if result == 1 else 'Fail'}")

# 4. EXPLANATION:
# The model does not need to be trained again because joblib.dump() saved what
# the model already learned (its coefficients and intercept) into a file.
# joblib.load() restores that same fitted model, so predict() can use it right away.