
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# 1. Sample student dataset
data = {
    "study_hours": [
        1.0, 1.5, 2.0, 2.5, 3.0,
        3.5, 4.0, 5.0, 5.5, 6.0,
        6.5, 7.0, 8.0, 8.5, 9.0
    ],
    "attendance_percentage": [
        50, 55, 60, 65, 70,
        60, 75, 80, 70, 85,
        90, 85, 90, 95, 98
    ],
    "passed": [
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# 2. Select the input features and target
X = df[["study_hours", "attendance_percentage"]]
y = df["passed"]

# 3. Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X, y)

# 4. Save the trained model
model_path = (
    Path(__file__).resolve().parent
    / "student_pass_model.joblib"
)

joblib.dump(model, model_path)

print("Model trained and saved successfully!")
print(f"Model location: {model_path}")

# 5. Load the saved model without retraining
loaded_model = joblib.load(model_path)

print("Saved model loaded successfully!")

# 6. Make three predictions using different study hours
test_students = [
    [2.0, 75],
    [5.0, 75],
    [8.0, 75]
]

print("\nThree Student Predictions:")

for student in test_students:
    hours, attendance = student

    # Use the same feature names as the training dataset
    student_data = pd.DataFrame(
        [student],
        columns=["study_hours", "attendance_percentage"]
    )

    prediction = loaded_model.predict(student_data)[0]

    result = "Pass" if prediction == 1 else "Fail"

    print(
        f"Study hours: {hours}, "
        f"Attendance: {attendance}%, "
        f"Prediction: {result}"
    )

# 7. Explain model persistence
print("\nExplanation of Model Persistence:")
print(
    "The model does not need to be trained again because "
    "its learned parameters were saved in the .joblib file."
)
print(
    "We can load the saved model using joblib.load() "
    "and reuse it to make new predictions."
)