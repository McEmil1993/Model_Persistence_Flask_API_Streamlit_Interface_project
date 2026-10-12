import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Columns: [study_hours, attendance_percentage]
X = np.array([
    [1, 50], [2, 60], [2, 90], [3, 55], [3, 70],
    [4, 65], [4, 85], [5, 60], [5, 80], [6, 75],
    [6, 90], [7, 70], [7, 95], [8, 85], [9, 80],
    [10, 90], [1, 95], [8, 50], [9, 60], [10, 100],
])
y = np.array([0, 0, 0, 0, 0,
              0, 0, 0, 1, 1,
              1, 1, 1, 1, 1,
              1, 0, 1, 1, 1])

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "student_pass_model_v2.joblib")
print("Saved student_pass_model_v2.joblib")