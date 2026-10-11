
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Training dataset
study_hours = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10
]).reshape(-1, 1)

# 0 = Fail, 1 = Pass
results = np.array([
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1
])

# Train the model
model = LogisticRegression()
model.fit(study_hours, results)

# Save trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")

# Make three predictions
for hours in [2, 5, 9]:
    prediction = loaded_model.predict([[hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours} | Prediction: {result}")
