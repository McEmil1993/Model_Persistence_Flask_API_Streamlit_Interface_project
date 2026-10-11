import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Exercise 1: train on study hours only, save under a new filename
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])  # study hours
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])                  # 0 = fail, 1 = pass

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")
print("Saved student_pass_model.joblib")

# --- Load and predict (no retraining) ---
loaded = joblib.load("student_pass_model.joblib")
for hours in [1.5, 4.5, 7]:
    pred = loaded.predict([[hours]])[0]
    prob = loaded.predict_proba([[hours]])[0][1]
    print(f"{hours} study hours -> {'Pass' if pred == 1 else 'Fail'} (P(pass)={prob:.2f})")

# Why no retraining? Training (fit) computes the model's coefficients and
# intercept. joblib saves those learned numbers; loading restores them, and
# predict() just applies them to new input.