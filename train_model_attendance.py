import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# columns: study_hours, attendance_percentage
X = np.array([
    [0.5, 40], [1, 55], [1.5, 60], [2, 50], [2.5, 70], [3, 65],
    [3.5, 55], [4, 80], [5, 75], [6, 90], [7, 85], [8, 95],
    [1, 95], [4, 45], [2, 90], [6, 50],
])
y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1])

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
joblib.dump(model, "student_pass_model_attendance.joblib")
print("Saved student_pass_model_attendance.joblib")
