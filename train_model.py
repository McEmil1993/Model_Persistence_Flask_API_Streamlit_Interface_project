
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate demonstration data
rng = np.random.default_rng(42)
number_of_students = 1000

study_hours = rng.uniform(0, 10, number_of_students)
attendance = rng.uniform(40, 100, number_of_students)

# Create sample results using BOTH features
score = (
    0.55 * study_hours
    + 0.045 * attendance
    + rng.normal(0, 0.8, number_of_students)
)

results = np.where(score >= 4.5, "PASS", "FAIL")

# Prepare the dataset
X = pd.DataFrame({
    "study_hours": study_hours,
    "attendance_percentage": attendance
})

y = results

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model saved successfully!")
