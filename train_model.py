from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

# Training data: study hours -> 0 = Fail, 1 = Pass
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
passed = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

model = LogisticRegression()
model.fit(study_hours, passed)

# Exercise 1: persist the trained model.
joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# Load the persisted model and make three predictions without retraining.
loaded_model = joblib.load(MODEL_PATH)

for hours in [2, 5, 8]:
    prediction = int(loaded_model.predict([[hours]])[0])
    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours} -> Prediction: {label}")

print("The model does not need to be trained again because its learned parameters were saved to disk with joblib.dump() and restored with joblib.load().")