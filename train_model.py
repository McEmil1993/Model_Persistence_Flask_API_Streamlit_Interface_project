# Exercise 1 - Model Persistence
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
FILE_HOURS = os.path.join(BASE, "student_pass_model.joblib")
FILE_HOURS_ATTENDANCE = os.path.join(BASE, "student_pass_model_v2.joblib")


def make_forest():
    # random_state keeps the result the same every time the script is run
    return RandomForestClassifier(n_estimators=100, random_state=42)


# ------------------ training data ------------------
students = pd.DataFrame({
    "study_hours": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10],
    "passed":      [0,   0, 0,   0, 0,   0, 1,   1, 1, 1, 1, 1, 1, 1],
})

students_v2 = pd.DataFrame({
    "study_hours": [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 1, 6],
    "attendance":  [40, 50, 60, 55, 65, 70, 60, 75, 80, 85, 90, 95, 95, 40],
    "passed":      [0,   0, 0,   0, 0,   1, 0, 1, 1, 1, 1, 1, 0, 0],
})

# ------------------ train ------------------
forest = make_forest()
forest.fit(students[["study_hours"]], students["passed"])

# Task 1: save the model using joblib.dump() (different filename)
joblib.dump(forest, FILE_HOURS)
print("Step 1: model saved  -> student_pass_model.joblib")

# Task 2: load the model using joblib.load()
reloaded = joblib.load(FILE_HOURS)
print("Step 2: model loaded <- student_pass_model.joblib")

# Task 3: three predictions with different study hours
print("Step 3: predictions")
for hrs in [1, 4, 8]:
    answer = reloaded.predict(pd.DataFrame({"study_hours": [hrs]}))[0]
    print(f"        {hrs} hour(s) of study -> {'PASS' if answer == 1 else 'FAIL'}")

# Task 4: explain why retraining is not needed
print("Step 4: The model does not need retraining because fit() already learned the")
print("        patterns and they are stored inside the model object. joblib.dump() put")
print("        them in a file; joblib.load() brings them back, ready for predict().")

# Model for Exercise 2 (study hours + attendance)
forest_v2 = make_forest()
forest_v2.fit(students_v2[["study_hours", "attendance"]], students_v2["passed"])
joblib.dump(forest_v2, FILE_HOURS_ATTENDANCE)
print("Step 5: model saved  -> student_pass_model_v2.joblib (hours + attendance)")