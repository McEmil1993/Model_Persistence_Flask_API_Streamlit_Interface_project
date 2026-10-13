import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data: study hours -> pass (1) / fail (0)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Save with a new filename
joblib.dump(model, "student_pass_model.joblib")
print("Model saved to student_pass_model.joblib")