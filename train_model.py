import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Toy dataset: study hours per day -> 0 = fail, 1 = pass
hours = np.array([[0.5], [1], [1.5], [2], [2.5], [3], [3.5], [4], [5], [6], [7], [8]])
passed = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(hours, passed)

joblib.dump(model, "student_pass_model.joblib")   # was a different filename before
print("Saved student_pass_model.joblib")
