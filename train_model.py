
import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample training dataset
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8]]
exam_results = [0, 0, 0, 0, 1, 1, 1, 1]

# Train the model
student_model = DecisionTreeClassifier(random_state=10)
student_model.fit(study_hours, exam_results)

# Save the trained model
joblib.dump(student_model, "caitom_pass_model.joblib")

print("Model training completed successfully.")
print("Trained model saved as caitom_pass_model.joblib")
