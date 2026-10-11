import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Training dummy data (Hours studied -> Pass (1) / Fail (0))
# Replace or retain with your existing training dataset
X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model under the requested filename
model_filename = "student_pass_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved successfully as '{model_filename}'.")

# 2. Reload the model
loaded_model = joblib.load(model_filename)
print("Model reloaded successfully.")

# 3. Make three predictions using different study hours
test_hours = np.array([[2.5], [5.0], [7.5]])
predictions = loaded_model.predict(test_hours)

for hours, pred in zip(test_hours.flatten(), predictions):
    outcome = "Pass" if pred == 1 else "Fail"
    print(f"Study Hours: {hours} -> Prediction: {outcome} ({pred})")