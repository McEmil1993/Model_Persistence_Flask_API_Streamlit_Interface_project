
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Sample training dataset
# Replace these sample records with your actual dataset if available.
data = {
    "study_hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 3, 4, 5, 6, 7, 8, 9, 2, 10
    ],
    "attendance_percentage": [
        50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
        60, 70, 75, 80, 85, 90, 95, 98, 50, 100
    ],
    "result": [
        "Fail", "Fail", "Fail", "Fail", "Fail",
        "Pass", "Pass", "Pass", "Pass", "Pass",
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Fail", "Pass"
    ]
}

df = pd.DataFrame(data)

# Use both features for training
X = df[["study_hours", "attendance_percentage"]]
y = df["result"]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained successfully!")
print("Model saved as student_pass_model.joblib")

# Test predictions using both features
test_students = pd.DataFrame({
    "study_hours": [2, 5, 8],
    "attendance_percentage": [55, 70, 90]
})

predictions = model.predict(test_students)

for i, prediction in enumerate(predictions):
    print(
        f"Study Hours: {test_students.iloc[i]['study_hours']}, "
        f"Attendance: {test_students.iloc[i]['attendance_percentage']}%, "
        f"Prediction: {prediction}"
    )
