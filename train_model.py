
import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample training data: study hours
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

# Target: 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")
