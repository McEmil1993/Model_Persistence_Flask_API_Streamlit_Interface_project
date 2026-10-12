import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Columns: [study_hours, attendance_percentage]
X = np.array([
    [1, 50], [2, 60], [2, 85], [3, 55], [3, 70], [4, 60], [4, 90], [5, 65],
    [5, 80], [6, 70], [6, 95], [7, 75], [7, 90], [8, 85], [3, 95], [1, 90],
])
# 0 = fail, 1 = pass
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "student_pass_model_attendance.joblib")
print("Model saved as student_pass_model_attendance.joblib")