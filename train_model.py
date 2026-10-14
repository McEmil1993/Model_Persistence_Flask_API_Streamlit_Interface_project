import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# ---------- Exercise 1: study-hours-only model ----------
hours = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
passed = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(hours, passed)

joblib.dump(model, "student_pass_model.joblib")          # save
print("Model saved to student_pass_model.joblib")

loaded = joblib.load("student_pass_model.joblib")        # load (no retraining)
print("Model loaded from student_pass_model.joblib")

for h in [2, 5, 8]:
    pred = loaded.predict([[h]])[0]
    prob = loaded.predict_proba([[h]])[0][1]
    print(f"Study hours = {h}: prediction = {'PASS' if pred == 1 else 'FAIL'} (pass probability = {prob:.2f})")

# ---------- Exercise 2: retrained model (hours + attendance) ----------
X2 = np.array([[1, 40], [2, 50], [3, 55], [4, 60], [5, 70], [6, 75],
               [7, 85], [8, 90], [9, 95], [10, 100], [8, 30], [2, 95],
               [5, 50], [6, 60], [4, 90], [3, 80]])
y2 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0])

model2 = LogisticRegression(max_iter=1000).fit(X2, y2)
joblib.dump(model2, "student_pass_model_v2.joblib")
print("Model v2 (hours + attendance) saved to student_pass_model_v2.joblib")