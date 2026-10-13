
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

# Save the model in the same folder as this Python file
MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

data = {
    "study_hours": [
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0
    ],
    "attendance_percentage": [
        50, 55, 60, 65, 70, 60, 75, 80,
        70, 85, 90, 85, 90, 95, 98
    ],
    "passed": [
        0, 0, 0, 0, 0, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance_percentage"]]
y = df["passed"]

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# Load the saved model
loaded_model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Make three predictions
test_data = pd.DataFrame({
    "study_hours": [2.0, 5.0, 8.0],
    "attendance_percentage": [60, 75, 95]
})

predictions = loaded_model.predict(test_data)

for i in range(len(test_data)):
    result = "Passed" if predictions[i] == 1 else "Failed"

    print(f"Study Hours: {test_data.iloc[i]['study_hours']}")
    print(
        f"Attendance: "
        f"{test_data.iloc[i]['attendance_percentage']}%"
    )
    print(f"Prediction: {result}")
    print("-" * 20)
