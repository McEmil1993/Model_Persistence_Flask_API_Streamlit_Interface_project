import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Columns: study_hours, attendance_percentage -> pass (1) / fail (0)
X = np.array([
    [1, 40], [2, 50], [2, 90], [3, 60], [3, 85],
    [4, 55], [4, 95], [5, 70], [5, 45], [6, 80],
    [7, 65], [7, 90], [8, 75], [9, 85], [10, 95],
    [1, 95], [6, 40], [8, 50], [9, 60], [10, 70],
])
y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "student_pass_model_v2.joblib")
print("Two-feature model saved to student_pass_model_v2.joblib")