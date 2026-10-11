"""
Train a simple student pass/fail model and save it for reuse.
Run: python train_model.py
"""
import joblib
from sklearn.tree import DecisionTreeClassifier

# Example teaching data: [study_hours] -> 0 = Fail, 1 = Pass
X = [[1], [2], [2.5], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved as student_pass_model.joblib")
