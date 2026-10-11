import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample training data
# Study hours and exam results
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = ["Fail", "Fail", "Fail", "Pass",
     "Pass", "Pass", "Pass", "Pass"]

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model saved successfully!")
