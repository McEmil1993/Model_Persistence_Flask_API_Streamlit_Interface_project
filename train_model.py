
import joblib
from sklearn.linear_model import LogisticRegression

# Sample training data: study hours and exam outcome
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = ["Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass"]

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
study_hours = [[2], [5], [7]]

for hours in study_hours:
    prediction = loaded_model.predict([hours])[0]
    print(f"Study hours: {hours[0]} | Prediction: {prediction}")
