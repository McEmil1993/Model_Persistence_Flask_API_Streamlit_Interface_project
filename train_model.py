import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Study hours -> pass (1) / fail (0)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Save with a different filename
joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved to student_pass_model.joblib")

# Load it back
loaded_model = joblib.load("student_pass_model.joblib")

# Three predictions with different study hours
for hours in [2, 5, 9]:
    result = loaded_model.predict([[hours]])[0]
    label = "PASS" if result == 1 else "FAIL"
    print(f"{hours} study hours -> {label}")

# Why no retraining is needed:
# Training (fit) is where the model learns its parameters (coefficients and
# intercept) from the data. joblib.dump stores those learned parameters in the
# file. joblib.load restores the same fitted model, so predict() can use the
# stored parameters straight away without calling fit() again.