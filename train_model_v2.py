import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Columns: study_hours, attendance_percentage
X = np.array([
    [1, 40], [2, 55], [3, 60], [4, 70],
    [5, 65], [6, 85], [7, 90], [8, 95],
    [2, 90], [6, 50],
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "student_pass_model_v2.joblib")
print("Saved student_pass_model_v2.joblib")