import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Study hours -> 0 = fail, 1 = pass
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Save the trained model under a new filename
joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved as student_pass_model.joblib")