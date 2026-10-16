
import joblib
from sklearn.linear_model import LogisticRegression

# Sample student data
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8]]
pass_results = [0, 0, 0, 0, 1, 1, 1, 1]

# Train the machine learning model
model = LogisticRegression()
model.fit(study_hours, pass_results)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Student Pass Prediction Model")
print("Model trained and saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")

# Make three predictions
test_hours = [[2], [5], [8]]

for hours in test_hours:
    prediction = loaded_model.predict(hours)
    result = "PASS" if prediction[0] == 1 else "FAIL"
    print(f"Study hours: {hours[0]} | Prediction: {result}")
