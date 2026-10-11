
import os
import joblib
from sklearn.tree import DecisionTreeClassifier

# Training data:
# Each row contains [study_hours, attendance_percentage]
X = [
    [1, 50],
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 75],
    [6, 80],
    [7, 90],
    [8, 95],
    [2, 40],
    [3, 50],
    [4, 55],
    [5, 60],
    [6, 65],
    [7, 75],
    [8, 85],
    [9, 98]
]

# 0 = Fail, 1 = Pass
y = [
    0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1, 1, 1
]

# Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the updated model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/student_pass_model.joblib"
)

print("Model trained successfully!")
print("Features: Study hours and attendance percentage.")
print("Saved to models/student_pass_model.joblib")
