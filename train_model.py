import joblib
from sklearn.linear_model import LogisticRegression

# Each row is [study_hours, attendance_percentage]
X = [[1, 40], [2, 50], [3, 55], [4, 60], [5, 65],
     [6, 70], [7, 75], [8, 80], [9, 90], [10, 95]]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass

model = LogisticRegression().fit(X, y)
joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved.")