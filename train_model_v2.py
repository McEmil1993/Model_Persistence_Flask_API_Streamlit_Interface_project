"""Retrains the model with TWO features so attendance can influence predictions
(used by Exercise 2's Flask API).

Features: [study_hours, attendance_percentage]  ->  0 = fail, 1 = pass
Saves to student_pass_model_v2.joblib
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_FILE_V2 = "student_pass_model_v2.joblib"

# [study_hours, attendance_percentage]
X = np.array([
    [0.5, 40], [1, 55], [1.5, 60], [2, 45], [2, 70], [2.5, 50],
    [3, 65], [3, 85], [3.5, 60], [4, 70], [4, 90], [4.5, 75],
    [5, 60], [5, 85], [5.5, 80], [6, 70], [6, 95], [7, 75],
    [7, 90], [8, 80], [8, 100], [9, 85], [10, 95], [1, 95],
    [2, 90], [6, 40], [8, 50], [9, 60], [0.5, 90], [4, 50],
])
y = np.array([
    0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 1,
    0, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0,
    0, 0, 1, 1, 0, 0,
])

if __name__ == "__main__":
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE_V2)
    print(f"Two-feature model saved to '{MODEL_FILE_V2}'")
    print(f"Training accuracy: {model.score(X, y):.2f}")
    print(f"Coefficients [hours, attendance]: {model.coef_[0]}")
