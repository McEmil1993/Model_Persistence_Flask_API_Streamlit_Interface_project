"""Exercise 1 - train and SAVE models with joblib.

Run:  python train_model.py
Creates:
  student_pass_model.joblib             (study_hours only)
  student_pass_model_attendance.joblib  (study_hours + attendance_percentage)
"""
import csv

import joblib
import numpy as np

from model_lib import LogisticModel

hours, attendance, passed = [], [], []
with open("data/students.csv") as f:
    for row in csv.DictReader(f):
        hours.append(float(row["study_hours"]))
        attendance.append(float(row["attendance_percentage"]))
        passed.append(int(row["passed"]))

hours = np.array(hours)
attendance = np.array(attendance)
passed = np.array(passed)

# Model 1: study hours only (Exercises 1 and 3)
model_hours = LogisticModel().fit(hours.reshape(-1, 1), passed)
joblib.dump(model_hours, "student_pass_model.joblib")
print("Saved student_pass_model.joblib")

# Model 2: study hours + attendance (Exercise 2)
model_both = LogisticModel().fit(np.column_stack([hours, attendance]), passed)
joblib.dump(model_both, "student_pass_model_attendance.joblib")
print("Saved student_pass_model_attendance.joblib")
