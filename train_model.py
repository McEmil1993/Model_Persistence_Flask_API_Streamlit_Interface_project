
import joblib
from sklearn.tree import DecisionTreeClassifier

# Training data: study hours and pass/fail result
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8]]
results = [0, 0, 0, 0, 1, 1, 1, 1]

# Create and train the machine learning model
model = DecisionTreeClassifier(random_state=42)
model.fit(study_hours, results)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")

# Make predictions for three different study hours
test_hours = [[2], [5], [7]]

for hours in test_hours:
    prediction = loaded_model.predict([hours])[0]
    status = "PASS" if prediction == 1 else "FAIL"
    print(f"Study hours: {hours[0]} -> Prediction: {status}")
