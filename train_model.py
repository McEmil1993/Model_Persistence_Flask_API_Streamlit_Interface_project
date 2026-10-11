import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_FILENAME = "student_pass_model.joblib"
FEATURES = ["study_hours", "attendance_percentage"]


def make_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0, 10, n)
    attendance = rng.uniform(30, 100, n)
    score = 0.6 * study_hours + 0.05 * attendance + rng.normal(0, 0.8, n)
    passed = (score > 6.0).astype(int)
    return pd.DataFrame({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "passed": passed,
    })


df = make_dataset()
X = df[FEATURES]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.2f}")

joblib.dump(model, MODEL_FILENAME)
print(f"Model saved to '{MODEL_FILENAME}'")