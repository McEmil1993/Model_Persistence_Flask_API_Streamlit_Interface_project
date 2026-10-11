import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample training data: study hours and result (0 = Fail, 1 = Pass)
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 1, 1, 1, 1, 1]

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")

# Make three predictions using different study hours
test_hours = [[2], [4], [7]]

for row in test_hours:
    prediction = loaded_model.predict([row])[0]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {row[0]} -> Prediction: {result}")

# Explain why retraining is not needed
print(
    "\nThe model does not need to be trained again because "
    "joblib.load() loads the trained model saved by joblib.dump(). "
    "The loaded model can make predictions without repeating training."
)