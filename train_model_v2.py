"""Retrain the model using BOTH study hours and attendance percentage (for Exercise 2)."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model_v2.joblib"

# Columns: [study_hours, attendance_percentage]
X = np.array([
    [1, 40], [2, 55], [2, 80], [3, 50], [3, 90],
    [4, 60], [4, 85], [5, 70], [5, 95], [6, 60],
    [6, 85], [7, 75], [8, 80], [8, 95], [9, 90], [10, 100],
    [1, 95], [2, 30], [7, 45], [9, 50],
])
# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1])

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, MODEL_FILE)
print(f"Saved {MODEL_FILE}")
