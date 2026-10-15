"""
Exercise 1 — Model Persistence

Trains a Logistic Regression model on student data with TWO features:
  - study_hours (hours per day)
  - attendance_percentage (0-100)

Saves it with joblib, reloads it, and makes predictions.
"""

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model.joblib")

# --------------- Training data ---------------
# Features: [study_hours, attendance_percentage]
# Label: 1 = pass, 0 = fail
# Rule of thumb in this data: students generally pass when they BOTH
# study enough AND attend enough. Either feature alone is not enough.
X_train = np.array([
    # Fail examples
    [1, 30], [2, 40], [3, 50], [4, 55],
    [2, 80], [3, 80],           # decent attendance but too little study
    [8, 30], [9, 35],           # lots of study but very low attendance
    [4, 50], [3, 60], [6, 45],  # one feature too low
    [1, 90],                    # very high attendance, almost no study
    # Pass examples
    [5, 70], [6, 75], [7, 80], [8, 75],
    [6, 65], [5, 65], [7, 65], [9, 80],
    [10, 70], [4, 90], [5, 80], [6, 60],
    [5, 60], [4, 70],
])
y_train = np.array(
    [0] * 12 + [1] * 14
)

# --------------- Train the model ---------------
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model trained successfully.")
print(f"  Training accuracy: {model.score(X_train, y_train):.2f}")
print(f"  Coefficients: study_hours={model.coef_[0][0]:.3f}, "
      f"attendance={model.coef_[0][1]:.3f}, intercept={model.intercept_[0]:.3f}")
print(f"  Expected features: {model.n_features_in_} "
      "(study_hours, attendance_percentage)")

# --------------- Save the model ---------------
joblib.dump(model, MODEL_PATH)
print(f"Model saved to '{MODEL_PATH}'.")

# --------------- Load the model ---------------
loaded_model = joblib.load(MODEL_PATH)
print(f"Model loaded from '{MODEL_PATH}'.")

# --------------- Make predictions ---------------
test_cases = [
    (2, 30),   # low study + low attendance -> Fail
    (6, 85),   # good study + good attendance -> Pass
    (9, 40),   # high study but low attendance -> borderline
    (2, 95),   # low study but very high attendance -> borderline
    (5, 65),   # around the decision boundary
]
for hours, attendance in test_cases:
    prediction = loaded_model.predict([[hours, attendance]])[0]
    proba = loaded_model.predict_proba([[hours, attendance]])[0][1]
    label = "Pass" if prediction == 1 else "Fail"
    print(f"  Study hours: {hours}, Attendance: {attendance}%  "
          f"->  Prediction: {label} (P(pass)={proba:.2f})")

# --------------- Explanation ---------------
print("\n--- Why no retraining is needed ---")
print(
    "The model's learned parameters (weights and bias) are stored inside\n"
    "the .joblib file. When we load the file, the model is fully\n"
    "reconstructed in memory with those same parameters, so it can\n"
    "make predictions immediately without retraining."
)
