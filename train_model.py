
import joblib
from sklearn.linear_model import LogisticRegression

# Training data
X = [[1], [2], [3], [4], [5],
     [6], [7], [8], [9], [10]]

y = [
    "Fail", "Fail", "Fail", "Fail", "Pass",
    "Pass", "Pass", "Pass", "Pass", "Pass"
]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")

# Make three predictions
for hours in [2, 5, 8]:
    result = loaded_model.predict([[hours]])[0]
    print(f"Study Hours: {hours} -> Prediction: {result}")
