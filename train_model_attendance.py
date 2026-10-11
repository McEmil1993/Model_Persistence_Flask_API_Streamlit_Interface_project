import joblib
from sklearn.linear_model import LogisticRegression

# [study_hours, attendance_percentage] -> 0 = Fail, 1 = Pass
X = [[1, 50], [2, 60], [3, 55], [4, 65], [5, 70],
     [6, 75], [7, 85], [8, 90], [9, 95], [10, 100],
     [2, 95], [8, 50]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "student_pass_model_attendance.joblib")
print("Saved student_pass_model_attendance.joblib")