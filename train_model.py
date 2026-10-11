"""Exercise 1 - Model Persistence (study hours only)."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data: study hours -> 0 = Fail, 1 = Pass
hours = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
passed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(hours, passed)

# SAVE the trained model (different filename, as required)
joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved as student_pass_model.joblib")

# LOAD it back - no retraining needed
loaded_model = joblib.load("student_pass_model.joblib")

# Three predictions with different study hours
for h in [2, 5.5, 8]:
    result = loaded_model.predict([[h]])[0]
    print(f"Study hours: {h} -> {'Pass' if result == 1 else 'Fail'}")
