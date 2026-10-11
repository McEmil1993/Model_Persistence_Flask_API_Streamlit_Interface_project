
from sklearn.tree import DecisionTreeClassifier
import joblib

# Sample student performance data
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8]]
results = [
    "Fail", "Fail", "Fail", "Pass",
    "Pass", "Pass", "Pass", "Pass"
]

# Train the model
model = DecisionTreeClassifier()
model.fit(study_hours, results)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")
