"""Exercise 2 - retrain with study hours AND attendance percentage."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# columns: study_hours, attendance_percentage
X = np.array([
    [1, 50], [2, 60], [3, 55], [4, 70], [5, 65],
    [6, 80], [7, 75], [8, 90], [9, 85], [10, 95],
    [2, 95], [8, 40], [4, 90], [6, 50], [5, 85], [7, 60],
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, "student_pass_attendance_model.joblib")
print("Saved student_pass_attendance_model.joblib")
