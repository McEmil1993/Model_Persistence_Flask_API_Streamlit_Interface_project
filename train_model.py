import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Sample training data (replace with your own dataset if provided)
data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":      [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
})

X = data[["study_hours"]]
y = data["passed"]

model = LogisticRegression()
model.fit(X, y)

# Save with a different filename
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")