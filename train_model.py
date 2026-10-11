
import joblib
from sklearn.linear_model import LogisticRegression

# Example training data
# Each row contains: [study_hours, attendance_percentage]
X = [
    [1, 50],
    [2, 55],
    [2, 70],
    [3, 65],
    [3, 80],
    [4, 75],
    [4, 90],
    [5, 85],
    [6, 90],
    [7, 95],
    [8, 98],
    [9, 100]
]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")
