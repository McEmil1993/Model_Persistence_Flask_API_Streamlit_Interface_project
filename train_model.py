from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Example training data: study hours and pass/fail outcome.
# Replace this sample with your instructor's dataset if one was provided.
data = pd.DataFrame({
    "study_hours": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8],
    "passed":      [0,   0,  0,   0,  0,   0,  1,   1,  1,   1,  1,   1, 1, 1]
})
data.to_csv(DATA_DIR / "student_data.csv", index=False)

X = data[["study_hours"]]
y = data["passed"]

model = LogisticRegression()
model.fit(X, y)

model_path = MODEL_DIR / "student_pass_model.joblib"
joblib.dump(model, model_path)
print(f"Model trained and saved to: {model_path}")

# Load the saved model and make three sample predictions.
loaded_model = joblib.load(model_path)
for hours in [1.5, 3.0, 5.0]:
    prediction = int(loaded_model.predict(pd.DataFrame({"study_hours": [hours]}))[0])
    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours:.1f} -> Prediction: {label}")

print("\nThe model does not need to be retrained for these predictions because")
print("joblib.load() restores the already-trained model and its learned parameters.")
