"""
Train a simple student pass prediction model and save it to disk.
The example dataset is illustrative; replace it with your class dataset if provided.
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

# Example training data: study hours and whether the student passed (1) or not (0).
study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1.5, 2.5, 3.5, 4.5]).reshape(-1, 1)
passed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1])

model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
model.fit(study_hours, passed)

joblib.dump(model, MODEL_PATH)
print(f"Trained model saved to: {MODEL_PATH.name}")

# Load the saved model and make three predictions without retraining.
loaded_model = joblib.load(MODEL_PATH)
for hours in [2, 5, 8]:
    prediction = int(loaded_model.predict([[hours]])[0])
    label = "Pass" if prediction == 1 else "Fail"
    probability = float(loaded_model.predict_proba([[hours]])[0][1])
    print(f"Study hours: {hours:>2} -> {label} (pass probability: {probability:.1%})")
