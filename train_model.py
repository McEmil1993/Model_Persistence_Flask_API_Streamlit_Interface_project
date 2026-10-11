
import joblib
from sklearn.linear_model import LogisticRegression

# Sample data: study hours and pass/fail results
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Train the model
model = LogisticRegression()
model.fit(X, y)

print("Model trained successfully!")

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")

# Load the saved model without retraining
loaded_model = joblib.load("student_pass_model.joblib")
print("Saved model loaded successfully!")

# Make three predictions
hours_to_predict = [[2], [5], [8]]
predictions = loaded_model.predict(hours_to_predict)

for hours, prediction in zip(hours_to_predict, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours[0]} -> Prediction: {result}")
