import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data with TWO features: study_hours and attendance_percentage
X = np.array([
    [1, 40], [2, 45], [3, 50], [4, 55], [5, 60],
    [5, 70], [6, 75], [7, 80], [8, 85], [9, 90], [10, 95]
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved to: student_pass_model.joblib")

# Load the model back (proves persistence)
loaded_model = joblib.load("student_pass_model.joblib")

# Three predictions with different study hours (attendance fixed for demo)
test_cases = [
    (2, 50),
    (5, 70),
    (8, 90)
]

print("\n--- Sample Predictions ---")
for hours, attendance in test_cases:
    pred = loaded_model.predict([[hours, attendance]])[0]
    proba = loaded_model.predict_proba([[hours, attendance]])[0][1]
    label = "Pass" if pred == 1 else "Fail"
    print(f"Study hours: {hours} | Attendance: {attendance}% | "
          f"Prediction: {label} | Pass probability: {proba:.1%}")

print(
    "\nRetraining is not needed for each prediction because "
    "joblib.load() restores the learned model parameters from the saved file."
)