# Exercise 2 - Retrain the model using study hours + attendance
import joblib
from sklearn.linear_model import LogisticRegression

# [study_hours, attendance_percentage]
X = [[1, 50], [2, 60], [3, 55], [4, 70], [5, 75], [6, 85], [7, 90], [8, 95]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "student_pass_model_v2.joblib")
print("Saved student_pass_model_v2.joblib")
