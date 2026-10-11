
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Sample student data
data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})

# Separate input and output
X = data[["study_hours"]]
y = data["passed"]

# Train the model
model = LogisticRegression()
model.fit(X, y)

print("Model trained successfully!")

# Save the model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")

# Make three predictions
test_data = pd.DataFrame({
    "study_hours": [2, 5, 8]
})

predictions = loaded_model.predict(test_data)

for hours, prediction in zip(test_data["study_hours"], predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours} -> Prediction: {result}")