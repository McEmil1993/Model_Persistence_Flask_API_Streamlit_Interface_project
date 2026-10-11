import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Sample data: study hours -> passed (1) or failed (0)
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":      [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
}
df = pd.DataFrame(data)

X = df[["study_hours"]]
y = df["passed"]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")