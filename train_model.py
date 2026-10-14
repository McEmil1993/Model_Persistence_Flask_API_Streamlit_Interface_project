import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data: study hours -> 0 = Fail, 1 = Pass
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model with a different filename
joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved as student_pass_model.joblib")

# Load the model back (no retraining)
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded from student_pass_model.joblib\n")

# Three predictions with different study hours
for hours in [2, 5.5, 9]:
    prediction = loaded_model.predict(np.array([[hours]]))[0]
    probability = loaded_model.predict_proba(np.array([[hours]]))[0][1]
    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours:>4} -> {label} (pass probability: {probability:.2f})")