# train_model.py (Exercise 1 + attendance)

from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# Sample data with 2 features: study_hours and attendance
X = np.array([
    [1, 40], [2, 50], [3, 55], [4, 60],
    [5, 70], [6, 75], [7, 80], [8, 85],
    [9, 90], [10, 95]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")

# Load the model
loaded_model = joblib.load("student_pass_model.joblib")

# Make 3 predictions
test_data = [
    [2, 50],
    [5.5, 75],
    [9, 90]
]

print("\nPredictions:")
for hours, attendance in test_data:
    pred = loaded_model.predict(np.array([[hours, attendance]]))[0]
    result = "Pass" if pred == 1 else "Fail"
    print(f"Study hours: {hours}, Attendance: {attendance}% → Prediction: {result}")

# Explanation required by Exercise 1
print("\nWhy the model does not need to be trained again:")
print("Because the trained model was saved into a file (student_pass_model.joblib).")
print("When we load the file, we can directly use the learned patterns to make predictions.")
print("There is no need to train the model again.")