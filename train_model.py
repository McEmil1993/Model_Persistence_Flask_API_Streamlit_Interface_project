
import joblib
from sklearn.linear_model import LogisticRegression

# Training data: study hours
X = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Results: 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")
