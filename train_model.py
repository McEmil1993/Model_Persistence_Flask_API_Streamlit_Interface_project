import joblib
from sklearn.tree import DecisionTreeClassifier

# Sample training data
# [study_hours, attendance]
X = [
    [1, 50],
    [2, 60],
    [3, 70],
    [4, 80],
    [5, 90],
    [6, 95]
]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 1, 1, 1]

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model saved successfully!")