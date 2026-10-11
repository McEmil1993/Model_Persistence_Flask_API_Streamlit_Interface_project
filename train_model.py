from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "student_pass_model.joblib"

data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed": [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
})

X = data[["study_hours"]]
y = data["passed"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")