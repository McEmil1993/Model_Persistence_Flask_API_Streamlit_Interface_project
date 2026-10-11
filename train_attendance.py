import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Exercise 2: retrain with two features [study_hours, attendance_percentage]
X = np.array([
    [1, 50], [2, 60], [3, 55], [4, 70], [2, 90], [3, 85],
    [5, 60], [5, 80], [6, 75], [7, 90], [8, 95], [4, 95],
])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
joblib.dump(model, "student_pass_model_attendance.joblib")
print("Saved student_pass_model_attendance.joblib")