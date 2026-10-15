
import joblib
from sklearn.linear_model import LogisticRegression

# Sample training data
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
results = [
    "Fail", "Fail", "Fail", "Fail", "Pass",
    "Pass", "Pass", "Pass", "Pass", "Pass"
]

# Train the model
model = LogisticRegression()
model.fit(study_hours, results)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")

# Make three predictions
test_hours = [[2], [5], [8]]

for hours in test_hours:
    prediction = loaded_model.predict([hours])[0]
    print(f"Study Hours: {hours[0]} - Prediction: {prediction}")
