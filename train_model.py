import joblib
from sklearn.linear_model import LogisticRegression

# 1. Prepare training data: [Study Hours, Attendance Percentage] -> [Fail (0) or Pass (1)]
X_train = [[1, 40], [2, 50], [3, 60], [5, 80], [6, 90], [8, 95]]
y_train = [0, 0, 0, 1, 1, 1]

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 2. Save the model using joblib.dump()
model_filename = 'student_pass_model.joblib'
joblib.dump(model, model_filename)
print(f"Model successfully saved to {model_filename}")

# 3. Load the model using joblib.load()
loaded_model = joblib.load(model_filename)
print("Model successfully loaded from disk.")

# 4. Make three predictions using different study hours and attendance
test_cases = [[2, 55], [4, 75], [7, 92]]
predictions = loaded_model.predict(test_cases)

print("\n--- Predictions ---")
for hours_attendance, pred in zip(test_cases, predictions):
    status = "Pass" if pred == 1 else "Fail"
    print(f"Study Hours: {hours_attendance[0]}, Attendance: {hours_attendance[1]}% -> Prediction: {status}")

# 5. Explanation: Why the model doesn't need to be retrained
print("\n--- Explanation ---")
print("Once trained, the model's internal parameters (weights and bias) are finalized")
print("and stored in the file. joblib.load() restores these exact parameters into memory,")
print("allowing instant predictions via matrix multiplication without re-running training loops.")