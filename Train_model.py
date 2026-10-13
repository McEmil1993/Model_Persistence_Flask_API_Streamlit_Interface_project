
import joblib
from sklearn.linear_model import LogisticRegression

# Study hours used for training
X = [[1], [2], [3], [4], [5], [6], [7], [8]]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Create the machine learning model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")
