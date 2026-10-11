
import joblib
from sklearn.tree import DecisionTreeClassifier

# Training data: number of study hours
hours_data = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Labels: 0 = Fail, 1 = Pass
result_labels = [0, 0, 0, 0, 1, 1, 1, 1]

# Create and train the model
tree_model = DecisionTreeClassifier()
tree_model.fit(hours_data, result_labels)

# Save the trained model
joblib.dump(tree_model, "student_pass_model.joblib")

print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")

# Test predictions using three study-hour values
sample_hours = [[2], [5], [7]]
results = loaded_model.predict(sample_hours)

# Display the predictions
for study, label in zip(sample_hours, results):
    prediction_text = "Pass" if label == 1 else "Fail"
    print(f"Study hours: {study[0]} - Prediction: {prediction_text}")