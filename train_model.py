
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "pass": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["study_hours"]]
y = df["pass"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "student_pass_model.joblib")

print("Model trained and saved successfully!")
