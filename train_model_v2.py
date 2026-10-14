import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Columns: study_hours, attendance_percentage
X = np.array([
    [1, 50], [2, 60], [3, 55], [4, 65], [5, 70],
    [6, 75], [7, 80], [8, 85], [9, 90], [10, 95],
    [2, 95], [4, 90], [6, 50], [8, 60], [3, 90], [7, 55],
])
# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1])

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "student_pass_model_v2.joblib")
print("Saved student_pass_model_v2.joblib")