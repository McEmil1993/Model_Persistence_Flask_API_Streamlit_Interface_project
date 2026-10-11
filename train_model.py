
import joblib
from sklearn.linear_model import LogisticRegression

# Sample student data: study hours
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, "student_pass_model.joblib")

print("Model trained successfully!")
print("Model saved as student_pass_model.joblib")
