
import joblib
from sklearn.tree import DecisionTreeClassifier

# Training data: number of study hours
study_data = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Labels: 0 = Fail, 1 = Pass
student_results = [0, 0, 0, 0, 1, 1, 1, 1]

# Create and train the model
classifier = DecisionTreeClassifier()
classifier.fit(study_data, student_results)

# Save the trained model
joblib.dump(classifier, "student_pass_model.joblib")

print("Model saved successfully!")

# Load the saved model
saved_classifier = joblib.load("student_pass_model.joblib")

# Test predictions using three study-hour values
test_hours = [[2], [5], [7]]
predicted_results = saved_classifier.predict(test_hours)

# Display the predictions
for hour_data, predicted_label in zip(test_hours, predicted_results):
    status = "Pass" if predicted_label == 1 else "Fail"
    print(f"Study hours: {hour_data[0]} - Prediction: {status}")