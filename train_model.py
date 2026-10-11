
import joblib
from sklearn.neighbors import KNeighborsClassifier

# Sample student data
# Input: study hours
study_hours = [
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
]

# Output: 0 = Fail, 1 = Pass
results = [
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 1
]

# Create and train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(study_hours, results)

# Save the trained model
filename = "student_pass_model.joblib"
joblib.dump(model, filename)

print(f"Model saved as {filename}")

# Load the saved model
loaded_model = joblib.load(filename)
print("Model loaded successfully!")

# Make three predictions using different study hours
test_hours = [[2], [5], [8]]
predictions = loaded_model.predict(test_hours)

for hours, prediction in zip(test_hours, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours[0]} -> Prediction: {result}")
