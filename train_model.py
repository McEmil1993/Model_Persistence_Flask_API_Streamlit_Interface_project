import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# SAMPLE DATA FOR THE SCHOOL ACTIVITY
# Each row contains:
# study_hours, attendance_percentage, final_result
# These are illustrative examples, not real student records.

data = {
    "study_hours": [
        1, 2, 1, 3, 2, 4, 3, 5, 4, 6,
        2, 5, 6, 7, 4, 8, 3, 7, 9, 5,
        1, 6, 8, 2, 7, 4, 9, 3, 8, 6,
        5, 10, 2, 7, 4, 9, 1, 6, 8, 5
    ],
    "attendance_percentage": [
        40, 50, 85, 60, 95, 80, 90, 75, 95, 90,
        30, 95, 70, 85, 65, 95, 45, 60, 90, 80,
        95, 55, 70, 75, 95, 85, 60, 95, 85, 40,
        90, 95, 55, 70, 50, 80, 35, 95, 55, 65
    ],
    "result": [
        "Fail", "Fail", "Fail", "Fail", "Fail",
        "Pass", "Pass", "Pass", "Pass", "Pass",
        "Fail", "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Pass", "Pass",
        "Fail", "Fail", "Pass", "Fail", "Pass",
        "Pass", "Pass", "Fail", "Pass", "Fail",
        "Pass", "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Pass", "Pass"
    ]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Input features and target label
X = df[["study_hours", "attendance_percentage"]]
y = df["result"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create and train the machine learning model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42, max_iter=1000)
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Model training completed!")
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("\nModel saved as student_pass_model.joblib")

# Test three sample students
test_students = pd.DataFrame({
    "study_hours": [2, 5, 8],
    "attendance_percentage": [40, 75, 95]
})

print("\nSample Predictions:")
for i, row in test_students.iterrows():
    result = model.predict(
        pd.DataFrame([row], columns=X.columns)
    )[0]

    print(
        f"Study Hours: {row['study_hours']}, "
        f"Attendance: {row['attendance_percentage']}%, "
        f"Prediction: {result}"
    )