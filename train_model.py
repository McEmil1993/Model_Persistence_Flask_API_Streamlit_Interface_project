import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])   # study hours
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])                   # 0 = fail, 1 = pass

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")
