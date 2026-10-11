
import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample training data: study hours and pass/fail result
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass"]

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the trained model using a new filename
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully.")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Saved model loaded successfully.")

# Make three predictions using different study hours
study_hours = [[2], [5], [8]]

for hours in study_hours:
    prediction = loaded_model.predict([hours])[0]
    print(f"Study hours: {hours[0]} -> Prediction: {prediction}")
