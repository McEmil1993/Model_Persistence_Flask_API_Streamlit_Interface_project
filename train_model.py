import joblib
from sklearn.linear_model import LogisticRegression

# Features: [study_hours, attendance_percentage]
X = [[1, 40], [2, 50], [3, 60], [4, 65],
     [5, 75], [6, 85], [7, 90], [8, 95]]
y = [0, 0, 0, 0, 1, 1, 1, 1]  

model = LogisticRegression().fit(X, y)

# Exercise 1: save with a different filename
joblib.dump(model, "student_pass_model.joblib")
print("Model saved to student_pass_model.joblib")
