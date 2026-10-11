import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Features: [study_hours, attendance_percentage]
X = np.array([
    [1, 40], [2, 50], [3, 55], [4, 60],
    [5, 65], [6, 70], [7, 75], [8, 80],
    [9, 90], [10, 95]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, "student_pass_model.joblib")
print("Retrained with 2 features and saved.")