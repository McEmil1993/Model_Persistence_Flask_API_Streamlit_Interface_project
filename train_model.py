import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample training data
# X = study hours
# y = student result
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [
    "Fail",
    "Fail",
    "Fail",
    "Pass",
    "Pass",
    "Pass",
    "Pass",
    "Pass",
]

# Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Saved model loaded successfully!")

# Make three predictions without retraining
test_hours = [2, 5, 7]

for hours in test_hours:
    prediction = loaded_model.predict([[hours]])[0]
    print(f"Study hours: {hours} -> Prediction: {prediction}")