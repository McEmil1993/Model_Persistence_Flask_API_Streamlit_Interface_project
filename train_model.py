import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample training data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array(["Fail", "Fail", "Fail", "Fail",
              "Pass", "Pass", "Pass", "Pass"])

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load model
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")

# Three predictions
prediction1 = loaded_model.predict([[2]])
prediction2 = loaded_model.predict([[5]])
prediction3 = loaded_model.predict([[8]])

print("2 study hours:", prediction1[0])
print("5 study hours:", prediction2[0])
print("8 study hours:", prediction3[0])