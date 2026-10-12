from sklearn.linear_model import LogisticRegression
import joblib

# Training data: study hours per day
study_hours = [
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
]

# Expected student results
results = [
    "Fail", "Fail", "Fail", "Fail", "Pass",
    "Pass", "Pass", "Pass", "Pass", "Pass"
]

# Create the machine learning model
model = LogisticRegression()

# Train the model
model.fit(study_hours, results)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained successfully!")
print("Model saved as student_pass_model.joblib")