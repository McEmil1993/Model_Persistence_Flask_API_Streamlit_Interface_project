import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# study_hours -> 0 = Fail, 1 = Pass
X_hours = np.array([[0.5], [1], [1.5], [2], [2.5], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X_hours, y)

joblib.dump(model, "student_pass_model.joblib")
print("Saved: student_pass_model.joblib")

loaded_model = joblib.load("student_pass_model.joblib")
print("Loaded: student_pass_model.joblib")

for hours in [1, 4, 8]:
    result = loaded_model.predict([[hours]])[0]
    print(f"{hours} hour(s) of study -> {'PASS' if result == 1 else 'FAIL'}")

X_both = np.array([
    [0.5, 40], [1, 50], [1.5, 60], [2, 55], [2.5, 65], [3, 70],
    [4, 60], [5, 75], [6, 80], [7, 85], [8, 90], [9, 95],
    [1, 95], [6, 40],
])
y_both = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0])

model_both = LogisticRegression()
model_both.fit(X_both, y_both)
joblib.dump(model_both, "student_pass_model_v2.joblib")
print("Saved: student_pass_model_v2.joblib (study hours + attendance)")