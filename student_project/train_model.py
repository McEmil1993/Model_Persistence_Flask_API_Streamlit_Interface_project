"""Exercise 1 - Model persistence (numpy-only version for Termux).

Saves two models with joblib:
  1. student_pass_model.joblib             -> study hours only
  2. student_pass_model_attendance.joblib  -> study hours + attendance (Exercise 2)
"""
import joblib
import numpy as np
from model_utils import LogisticModel

MODEL_FILE = "student_pass_model.joblib"                  # NEW filename
MODEL_ATT_FILE = "student_pass_model_attendance.joblib"

hours      = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10])
attendance = np.array([40, 50, 45, 60, 55, 70, 65, 75, 80, 85, 90, 88, 95, 98])
passed     = np.array([0,   0, 0,   0, 0,   0, 1,   0, 1,  1,  1,  1,  1,  1])

# Model 1: study hours only
model = LogisticModel().fit(hours.reshape(-1, 1), passed)
joblib.dump(model, MODEL_FILE)                 # save
print(f"Model saved to {MODEL_FILE}")

# Model 2: study hours + attendance
model_att = LogisticModel().fit(np.column_stack([hours, attendance]), passed)
joblib.dump(model_att, MODEL_ATT_FILE)
print(f"Model saved to {MODEL_ATT_FILE}")

# Load the model back - NO retraining
loaded = joblib.load(MODEL_FILE)
print(f"Model loaded from {MODEL_FILE}\n")

# Three predictions with different study hours
for h in [1, 4, 8]:
    pred = loaded.predict([[h]])[0]
    prob = loaded.predict_proba([[h]])[0][1]
    print(f"Study hours = {h:>2} -> {'PASS' if pred == 1 else 'FAIL'} (pass probability {prob:.2f})")

print("""
Why no retraining is needed:
Training (fit) only learns the model's parameters (coefficients and intercept).
joblib.dump() stores those learned values in the file, and joblib.load() restores
the exact same fitted object. predict() just applies the stored parameters to new
input, so it is fast and gives the same results every time.
""")
