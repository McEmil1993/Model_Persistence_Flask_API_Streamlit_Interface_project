from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
FEATURES = ["study_hours", "attendance_percentage"]

data = pd.read_csv(BASE_DIR / "data" / "students.csv")
X, y = data[FEATURES], data["passed"]

# Keep 20% of the data hidden so we can honestly measure accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy on unseen test data: {accuracy:.0%}")

joblib.dump(model, BASE_DIR / "models" / "student_pass_model_v2.joblib")
print("Saved models/student_pass_model_v2.joblib")
