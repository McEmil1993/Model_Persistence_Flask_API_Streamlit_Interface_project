
import joblib
from sklearn.linear_model import LogisticRegression

# Sample training data
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "student_pass_model.joblib")

print("Model saved successfully!")
