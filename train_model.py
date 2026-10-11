
from sklearn.tree import DecisionTreeClassifier
import joblib

# Example training data: study hours
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8]]

# 0 = Fail, 1 = Pass
results = [0, 0, 0, 1, 1, 1, 1, 1]

# Create and train the model
model = DecisionTreeClassifier()
model.fit(study_hours, results)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")
