import joblib
from sklearn.ensemble import RandomForestClassifier

# Training data: [Study Hours, Previous Grades, Attendance Percentage] -> [Fail (0) or Pass (1)]
X_train = [
    [1, 50, 40],
    [2, 55, 50],
    [3, 60, 60],
    [4, 65, 70],
    [6, 75, 85],
    [7, 80, 90],
    [9, 90, 95]
]
y_train = [0, 0, 0, 0, 1, 1, 1]

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Save the model using joblib.dump()
model_filename = 'student_pass_model.joblib'
joblib.dump(model, model_filename)
print(f"Alternative model successfully saved to {model_filename}")

# Load the model using joblib.load()
loaded_model = joblib.load(model_filename)
print("Alternative model successfully loaded from disk.")

# Make three predictions using different test cases
test_cases = [[2, 52, 45], [5, 70, 78], [8, 88, 92]]
predictions = loaded_model.predict(test_cases)

print("\n--- Alternative Predictions ---")
for inputs, pred in zip(test_cases, predictions):
    status = "Pass" if pred == 1 else "Fail"
    print(f"Hours: {inputs[0]}, Grade: {inputs[1]}, Attendance: {inputs[2]}% -> Prediction: {status}")

# Explanation
print("\n--- Explanation ---")
print("The model's internal tree structures and split criteria are saved to the file.")
print("joblib.load() restores these structures into memory, bypassing the training phase entirely.")