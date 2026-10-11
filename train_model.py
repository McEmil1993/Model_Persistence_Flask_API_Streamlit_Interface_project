# Exercise 1 - Train and save the model (study hours only)
import joblib
from sklearn.linear_model import LogisticRegression

# Study hours -> pass (1) / fail (0)
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)

# Save using a different filename
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")
