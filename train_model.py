
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

# Example training data: study hours per day and the class label.
# 0 = Fail, 1 = Pass. Replace this sample data with your instructor's dataset if provided.
study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
passed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(study_hours, passed)

joblib.dump(model, MODEL_PATH)
print(f"Model trained and saved to: {MODEL_PATH}")

# Load the saved model and make three predictions without retraining.
loaded_model = joblib.load(MODEL_PATH)
for hours in [2, 5, 8]:
    prediction = int(loaded_model.predict([[hours]])[0])
    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours} -> Prediction: {label}")
