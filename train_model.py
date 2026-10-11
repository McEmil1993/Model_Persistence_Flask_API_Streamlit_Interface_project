# train_model.py
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample training data: [study_hours, attendance_percentage]
X = np.array([
    [1, 20], [2, 35], [2, 60], [3, 40], [3, 75],
    [4, 50], [4, 85], [5, 55], [5, 90], [6, 65],
    [7, 70], [8, 80],
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])  # 0=Fail, 1=Pass

model = LogisticRegression()
model.fit(X, y)

# SAVE with custom name
joblib.dump(model, "student_pass_model.joblib")
print("Model saved!")

# LOAD it back
loaded_model = joblib.load("student_pass_model.joblib")

# 3 predictions
for hours, attendance in [(2, 40), (5, 70), (8, 90)]:
    pred = loaded_model.predict([[hours, attendance]])[0]
    print(f"{hours} hours, {attendance}% attendance -> {'Pass' if pred == 1 else 'Fail'}")
