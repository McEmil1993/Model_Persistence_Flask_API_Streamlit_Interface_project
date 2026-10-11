
import joblib
from sklearn.linear_model import LogisticRegression

# Simple training data: study hours vs pass/fail
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved as student_pass_model.joblib")
