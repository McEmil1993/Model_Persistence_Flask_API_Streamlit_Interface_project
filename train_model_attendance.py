# Exercise 2 - Retrain the model with study hours + attendance percentage
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# columns: [study_hours, attendance_percentage]
X = np.array([
    [1, 50], [2, 60], [2, 90], [3, 55], [3, 85], [4, 70], [4, 95],
    [5, 60], [5, 80], [6, 75], [6, 90], [7, 65], [7, 85], [8, 70],
    [8, 95], [9, 80], [10, 90], [1, 95], [9, 55], [3, 40],
])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "student_pass_model_attendance.joblib")
print("Model saved as student_pass_model_attendance.joblib")
